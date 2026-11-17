// bm_verif: TWO-BLOCK X2 engine. The flow-step-4 engine (x2_engine.v) on both RERAM_CIM_32x32 blocks of the chip, so one
// block can be programmed while the other computes (the schedules compiler/x2_schedule.py found faster).
//
// Host pins (35 of 36): clk, rst, cmd_data[11:0], cmd_hi, cmd_strobe, cmd_ready[1:0], acc_clear, idle[1:0],
//   acc_sel[3:0], byte_sel[1:0] -> acc_byte[7:0].
// Commands: two strobes as before (low 12 bits, then the high beat). The high beat carries
//   cmd_data[9:0] = command bits 21:12, cmd_data[10] = input slice of this command, cmd_data[11] = block (0/1).
//   cmd_ready[b] = block b can take a command (a host may keep feeding one block while the other is busy).
// Results: both blocks may finish a column in the same clock; a round-robin arbiter grants one (RESULT_READY holds the
//   other: the BM model keeps o_result_valid/o_result_data stable until i_result_ready). The slice of a result is the
//   slice of the command its block accepted last (the model takes no command while a result is pending).
// Accumulators (3-stage pipeline: decode+shift, read, add+write): one bank of NACC per block. acc_sel[3] picks the bank (for reading and for acc_clear), acc_sel[2:0]
//   the accumulator. idle[b] = block b has no command pending and no result in flight: safe to read or clear bank b.
// The column rule is the one of x2_engine.v (accumulator = column / CPW, sign and shift from column % CPW).
module x2_engine2 #(
    parameter integer CELL_BITS = 1,
    parameter [15:0] TARGET_SET1 = 16'h6472, parameter [15:0] TARGET_SET2 = 16'h0003,
    parameter [15:0] TARGET_RESET1 = 16'h000B, parameter [15:0] TARGET_RESET2 = 16'h462B,
    parameter [9:0]  NO_OF_CLK_CYCLES = 10'd5, parameter [9:0] COUNTER_VALUE = 10'd5,
    parameter [6:0]  TDC_TIME_OUT = 7'd63, parameter [1:0] TDC_DEAD_TIME = 2'd2
) (
    input  wire        clk,
    input  wire        rst,
    input  wire [11:0] cmd_data,
    input  wire        cmd_hi,
    input  wire        cmd_strobe,
    output wire [1:0]  cmd_ready,
    input  wire        acc_clear,
    output wire [1:0]  idle,
    input  wire [3:0]  acc_sel,
    input  wire [1:0]  byte_sel,
    output wire [7:0]  acc_byte
);
    localparam integer CPW   = 2 * ((8 + CELL_BITS - 1) / CELL_BITS);
    localparam integer HALF  = CPW / 2;
    localparam integer NACC  = 32 / CPW;
    localparam integer SPLIT = (CELL_BITS > 1) ? 4 : 8;

    // ---- command side: a shared low half, one pending command per block
    reg  [11:0] lo;
    reg  [21:0] word0, word1;
    reg         pend0, pend1, slc_cmd0, slc_cmd1, slc_res0, slc_res1;
    wire [1:0]  x2_ready;
    wire        hi_load = cmd_strobe && cmd_hi;
    always @(posedge clk) begin
        if (rst) begin
            pend0 <= 1'b0; pend1 <= 1'b0;
        end else begin
            if (cmd_strobe && !cmd_hi) lo <= cmd_data;
            if (hi_load && !cmd_data[11] && !pend0) begin word0 <= {cmd_data[9:0], lo}; slc_cmd0 <= cmd_data[10]; pend0 <= 1'b1; end
            else if (pend0 && x2_ready[0])         begin pend0 <= 1'b0; slc_res0 <= slc_cmd0; end
            if (hi_load &&  cmd_data[11] && !pend1) begin word1 <= {cmd_data[9:0], lo}; slc_cmd1 <= cmd_data[10]; pend1 <= 1'b1; end
            else if (pend1 && x2_ready[1])         begin pend1 <= 1'b0; slc_res1 <= slc_cmd1; end
        end
    end
    assign cmd_ready = {!pend1, !pend0};

    // ---- result side: round-robin grant, then the two-stage accumulate of x2_engine.v with a block bit
    wire [1:0]  rv;
    wire [20:0] rd0, rd1;
    reg         last_g1;
    wire        g0 = rv[0] && (!rv[1] || last_g1);
    wire        g1 = rv[1] && !g0;
    always @(posedge clk) if (rst) last_g1 <= 1'b0; else if (g0 || g1) last_g1 <= g1;
    wire [20:0] res     = g1 ? rd1 : rd0;
    wire        r_slice = g1 ? slc_res1 : slc_res0;
    wire [1:0]  r_mode  = res[20:19];
    wire [4:0]  r_col   = res[18:14];
    wire [13:0] r_count = res[13:0];
    wire [4:0]  slot    = r_col % CPW;
    wire        r_neg   = (slot >= HALF);
    wire [3:0]  r_shift = (slot % HALF) * CELL_BITS + (r_slice ? SPLIT : 0);
    wire [2:0]  r_acc   = r_col / CPW;
    reg         s1_valid, s1_blk, s1_neg;
    reg  [2:0]  s1_acc;
    reg  [31:0] s1_value;
    wire        clr0 = acc_clear && !acc_sel[3], clr1 = acc_clear && acc_sel[3];
    always @(posedge clk) begin
        if (rst) s1_valid <= 1'b0;
        else begin
            s1_valid <= (g0 || g1) && (r_mode == 2'd2) && !(g1 ? clr1 : clr0);
            s1_blk   <= g1;
            s1_acc   <= r_acc;
            s1_neg   <= r_neg;
            s1_value <= {18'd0, r_count} << r_shift;
        end
    end
    // stage 2 reads the target accumulator into a register; stage 3 adds and writes. A result for the accumulator that stage 3
    // writes in the same clock is forwarded (consecutive columns share an accumulator), so no update is lost.
    reg  signed [31:0] acc [0:2*NACC-1];
    wire [3:0]  s1_idx = s1_blk * NACC + s1_acc;
    reg         s2_valid, s2_neg, s2_blk;
    reg  [3:0]  s2_idx;
    reg  [31:0] s2_value;
    reg  signed [31:0] s2_old;
    wire signed [31:0] nxt_full = s2_neg ? s2_old - $signed(s2_value) : s2_old + $signed(s2_value);
    wire        s2_writes = s2_valid && !(s2_blk ? clr1 : clr0);
    always @(posedge clk) begin
        if (rst) s2_valid <= 1'b0;
        else begin
            s2_valid <= s1_valid && !(s1_blk ? clr1 : clr0);
            s2_idx   <= s1_idx;
            s2_neg   <= s1_neg;
            s2_blk   <= s1_blk;
            s2_value <= s1_value;
            s2_old   <= (s2_writes && s2_idx == s1_idx) ? nxt_full : acc[s1_idx];
        end
    end
    integer i;
    always @(posedge clk) begin
        if (rst) begin
            for (i = 0; i < 2 * NACC; i = i + 1) acc[i] <= 32'sd0;
        end else begin
            for (i = 0; i < NACC; i = i + 1) begin
                if (clr0) acc[i] <= 32'sd0;
                if (clr1) acc[NACC + i] <= 32'sd0;
            end
            if (s2_writes) acc[s2_idx] <= nxt_full;
        end
    end
    wire [31:0] sel_acc = acc[acc_sel[3] * NACC + (acc_sel[2:0] % NACC)];
    assign acc_byte = sel_acc[8*byte_sel +: 8];
    assign idle = {!pend1 && !rv[1] && x2_ready[1] && !(s1_valid &&  s1_blk) && !(s2_valid &&  s2_blk),
                   !pend0 && !rv[0] && x2_ready[0] && !(s1_valid && !s1_blk) && !(s2_valid && !s2_blk)};

    // ---- the two X2 blocks
    (* keep *)
    RERAM_CIM_32x32 #(
        .TARGET_SET1(TARGET_SET1), .TARGET_SET2(TARGET_SET2), .TARGET_RESET1(TARGET_RESET1), .TARGET_RESET2(TARGET_RESET2),
        .NO_OF_CLK_CYCLES(NO_OF_CLK_CYCLES), .COUNTER_VALUE(COUNTER_VALUE), .TDC_TIME_OUT(TDC_TIME_OUT), .TDC_DEAD_TIME(TDC_DEAD_TIME)
    ) x2_0 (
        .CLK(clk), .RST(rst), .CMD_VALID(pend0), .CMD_READY(x2_ready[0]),
        .MODE0(word0[0]), .MODE1(word0[1]),
        .ROW_INDEX0(word0[2]), .ROW_INDEX1(word0[3]), .ROW_INDEX2(word0[4]), .ROW_INDEX3(word0[5]), .ROW_INDEX4(word0[6]),
        .COL_INDEX0(word0[7]), .COL_INDEX1(word0[8]), .COL_INDEX2(word0[9]), .COL_INDEX3(word0[10]), .COL_INDEX4(word0[11]),
        .PWM_DATA0(word0[12]), .PWM_DATA1(word0[13]), .PWM_DATA2(word0[14]), .PWM_DATA3(word0[15]),
        .PWM_DATA4(word0[16]), .PWM_DATA5(word0[17]), .PWM_DATA6(word0[18]), .PWM_DATA7(word0[19]),
        .COMPUTE_LAST(word0[20]), .COMPUTE_ALL_COLUMNS(word0[21]),
        .BUSY(), .COMMAND_ACCEPTED(), .COMPUTE_PENDING(), .ERROR0(), .ERROR1(), .ERROR_VALID(),
        .OPERATION_DONE(), .PROGRAM_READBACK_ACTIVE(),
        .RESULT_VALID(rv[0]), .RESULT_READY(g0),
        .RESULT_DATA0(rd0[0]), .RESULT_DATA1(rd0[1]), .RESULT_DATA2(rd0[2]), .RESULT_DATA3(rd0[3]),
        .RESULT_DATA4(rd0[4]), .RESULT_DATA5(rd0[5]), .RESULT_DATA6(rd0[6]), .RESULT_DATA7(rd0[7]),
        .RESULT_DATA8(rd0[8]), .RESULT_DATA9(rd0[9]), .RESULT_DATA10(rd0[10]), .RESULT_DATA11(rd0[11]),
        .RESULT_DATA12(rd0[12]), .RESULT_DATA13(rd0[13]), .RESULT_DATA14(rd0[14]), .RESULT_DATA15(rd0[15]),
        .RESULT_DATA16(rd0[16]), .RESULT_DATA17(rd0[17]), .RESULT_DATA18(rd0[18]), .RESULT_DATA19(rd0[19]),
        .RESULT_DATA20(rd0[20]), .RESULT_LAST()
    );
    (* keep *)
    RERAM_CIM_32x32 #(
        .TARGET_SET1(TARGET_SET1), .TARGET_SET2(TARGET_SET2), .TARGET_RESET1(TARGET_RESET1), .TARGET_RESET2(TARGET_RESET2),
        .NO_OF_CLK_CYCLES(NO_OF_CLK_CYCLES), .COUNTER_VALUE(COUNTER_VALUE), .TDC_TIME_OUT(TDC_TIME_OUT), .TDC_DEAD_TIME(TDC_DEAD_TIME)
    ) x2_1 (
        .CLK(clk), .RST(rst), .CMD_VALID(pend1), .CMD_READY(x2_ready[1]),
        .MODE0(word1[0]), .MODE1(word1[1]),
        .ROW_INDEX0(word1[2]), .ROW_INDEX1(word1[3]), .ROW_INDEX2(word1[4]), .ROW_INDEX3(word1[5]), .ROW_INDEX4(word1[6]),
        .COL_INDEX0(word1[7]), .COL_INDEX1(word1[8]), .COL_INDEX2(word1[9]), .COL_INDEX3(word1[10]), .COL_INDEX4(word1[11]),
        .PWM_DATA0(word1[12]), .PWM_DATA1(word1[13]), .PWM_DATA2(word1[14]), .PWM_DATA3(word1[15]),
        .PWM_DATA4(word1[16]), .PWM_DATA5(word1[17]), .PWM_DATA6(word1[18]), .PWM_DATA7(word1[19]),
        .COMPUTE_LAST(word1[20]), .COMPUTE_ALL_COLUMNS(word1[21]),
        .BUSY(), .COMMAND_ACCEPTED(), .COMPUTE_PENDING(), .ERROR0(), .ERROR1(), .ERROR_VALID(),
        .OPERATION_DONE(), .PROGRAM_READBACK_ACTIVE(),
        .RESULT_VALID(rv[1]), .RESULT_READY(g1),
        .RESULT_DATA0(rd1[0]), .RESULT_DATA1(rd1[1]), .RESULT_DATA2(rd1[2]), .RESULT_DATA3(rd1[3]),
        .RESULT_DATA4(rd1[4]), .RESULT_DATA5(rd1[5]), .RESULT_DATA6(rd1[6]), .RESULT_DATA7(rd1[7]),
        .RESULT_DATA8(rd1[8]), .RESULT_DATA9(rd1[9]), .RESULT_DATA10(rd1[10]), .RESULT_DATA11(rd1[11]),
        .RESULT_DATA12(rd1[12]), .RESULT_DATA13(rd1[13]), .RESULT_DATA14(rd1[14]), .RESULT_DATA15(rd1[15]),
        .RESULT_DATA16(rd1[16]), .RESULT_DATA17(rd1[17]), .RESULT_DATA18(rd1[18]), .RESULT_DATA19(rd1[19]),
        .RESULT_DATA20(rd1[20]), .RESULT_LAST()
    );
`ifdef FORMAL
`include "x2_engine2_props.vh"
`endif
endmodule
