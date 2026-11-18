// x2c: one-block X2 engine = host bridge + recombiner in one fabric design.
// Same arithmetic as the x2_acc_step operator (acc +/- count << shift), split over two pipeline stages.
// The host streams the packed command stream (compiler/x2_pack.py); X2 COMPUTE results never leave the chip: each
// column count is added straight into an on-fabric 32-bit accumulator. The host reads finished partial outputs.
//
// Column roles come from the packing rule (every block starts on a multiple of CPW columns):
//   CPW = cells per weight = 2 * ceil(8 / CELL_BITS) (4 for 4-bit cells, 16 for 1-bit)
//   accumulator = column / CPW,  slot = column % CPW,  neg = slot >= CPW/2 (negative bank),
//   shift = (slot % (CPW/2)) * CELL_BITS + slice * SPLIT   (SPLIT = input slice bits: 4 for 4-bit cells, 8 for 1-bit)
// Row folds (a layer with more than 32 inputs per output) span several tiles; the host adds those partial outputs.
//
// Host pins (33 of 36): clk, rst, cmd_data[11:0], cmd_hi, cmd_strobe, cmd_ready (as in the host bridge),
//   slice (input slice of the next COMPUTE pass), acc_clear (zero all accumulators), idle (no command pending and no
//   result in flight: safe to read), acc_sel[2:0] + byte_sel[1:0] -> acc_byte[7:0].
// SET/RESET readbacks from the X2 are consumed and ignored; only COMPUTE results (mode 2) are accumulated.
module x2_engine #(
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
    output wire        cmd_ready,
    input  wire        slice,
    input  wire        acc_clear,
    output wire        idle,
    input  wire [2:0]  acc_sel,
    input  wire [1:0]  byte_sel,
    output wire [7:0]  acc_byte
);
    localparam integer CPW   = 2 * ((8 + CELL_BITS - 1) / CELL_BITS);
    localparam integer HALF  = CPW / 2;
    localparam integer NACC  = 32 / CPW;
    localparam integer SPLIT = (CELL_BITS > 1) ? 4 : 8;

    // ---- command side (same protocol as x2_bridge)
    reg [11:0] lo; reg [9:0] hi; reg pending;
    wire x2_cmd_ready, x2_res_valid, x2_res_last;
    wire [20:0] x2_res;
    wire [21:0] word = {hi, lo};
    always @(posedge clk) begin
        if (rst) pending <= 1'b0;
        else begin
            if (cmd_strobe && !cmd_hi && !pending) lo <= cmd_data;
            if (cmd_strobe &&  cmd_hi && !pending) begin hi <= cmd_data[9:0]; pending <= 1'b1; end
            else if (pending && x2_cmd_ready)       pending <= 1'b0;
        end
    end
    assign cmd_ready = !pending;

    // ---- result side: every result is consumed at once; COMPUTE results go into the accumulators.
    // Two pipeline stages (one result per clock still): stage 1 decodes the column and shifts the count,
    // stage 2 adds it into the selected accumulator. Splitting them shortens the longest path for timing.
    wire [1:0]  r_mode  = x2_res[20:19];
    wire [4:0]  r_col   = x2_res[18:14];
    wire [13:0] r_count = x2_res[13:0];
    wire [4:0]  slot    = r_col % CPW;
    wire        r_neg   = (slot >= HALF);
    wire [3:0]  r_shift = (slot % HALF) * CELL_BITS + (slice ? SPLIT : 0);
    wire [2:0]  r_acc   = r_col / CPW;

    reg         s1_valid;
    reg  [2:0]  s1_acc;
    reg         s1_neg;
    reg  [31:0] s1_value;                       // count << shift, already shifted
    always @(posedge clk) begin
        if (rst || acc_clear) s1_valid <= 1'b0;
        else begin
            s1_valid <= x2_res_valid && (r_mode == 2'd2);
            s1_acc   <= r_acc;
            s1_neg   <= r_neg;
            s1_value <= {18'd0, r_count} << r_shift;
        end
    end

    reg  signed [31:0] acc [0:NACC-1];
    wire signed [31:0] nxt_full = s1_neg ? acc[s1_acc] - $signed(s1_value) : acc[s1_acc] + $signed(s1_value);
    integer i;
    always @(posedge clk) begin
        if (rst || acc_clear) begin
            for (i = 0; i < NACC; i = i + 1) acc[i] <= 32'sd0;
        end else if (s1_valid) begin
            acc[s1_acc] <= nxt_full;
        end
    end

    wire [31:0] sel_acc = acc[acc_sel % NACC];
    assign acc_byte = sel_acc[8*byte_sel +: 8];
    assign idle = !pending && !x2_res_valid && !s1_valid && x2_cmd_ready;

    (* keep *)
    RERAM_CIM_32x32 #(
        .TARGET_SET1(TARGET_SET1), .TARGET_SET2(TARGET_SET2), .TARGET_RESET1(TARGET_RESET1), .TARGET_RESET2(TARGET_RESET2),
        .NO_OF_CLK_CYCLES(NO_OF_CLK_CYCLES), .COUNTER_VALUE(COUNTER_VALUE), .TDC_TIME_OUT(TDC_TIME_OUT), .TDC_DEAD_TIME(TDC_DEAD_TIME)
    ) x2 (
        .CLK(clk), .RST(rst),
        .CMD_VALID(pending), .CMD_READY(x2_cmd_ready),
        .MODE0(word[0]), .MODE1(word[1]),
        .ROW_INDEX0(word[2]), .ROW_INDEX1(word[3]), .ROW_INDEX2(word[4]), .ROW_INDEX3(word[5]), .ROW_INDEX4(word[6]),
        .COL_INDEX0(word[7]), .COL_INDEX1(word[8]), .COL_INDEX2(word[9]), .COL_INDEX3(word[10]), .COL_INDEX4(word[11]),
        .PWM_DATA0(word[12]), .PWM_DATA1(word[13]), .PWM_DATA2(word[14]), .PWM_DATA3(word[15]),
        .PWM_DATA4(word[16]), .PWM_DATA5(word[17]), .PWM_DATA6(word[18]), .PWM_DATA7(word[19]),
        .COMPUTE_LAST(word[20]), .COMPUTE_ALL_COLUMNS(word[21]),
        .BUSY(), .COMMAND_ACCEPTED(), .COMPUTE_PENDING(), .ERROR0(), .ERROR1(), .ERROR_VALID(),
        .OPERATION_DONE(), .PROGRAM_READBACK_ACTIVE(),
        .RESULT_VALID(x2_res_valid), .RESULT_READY(1'b1),
        .RESULT_DATA0(x2_res[0]), .RESULT_DATA1(x2_res[1]), .RESULT_DATA2(x2_res[2]), .RESULT_DATA3(x2_res[3]),
        .RESULT_DATA4(x2_res[4]), .RESULT_DATA5(x2_res[5]), .RESULT_DATA6(x2_res[6]), .RESULT_DATA7(x2_res[7]),
        .RESULT_DATA8(x2_res[8]), .RESULT_DATA9(x2_res[9]), .RESULT_DATA10(x2_res[10]), .RESULT_DATA11(x2_res[11]),
        .RESULT_DATA12(x2_res[12]), .RESULT_DATA13(x2_res[13]), .RESULT_DATA14(x2_res[14]), .RESULT_DATA15(x2_res[15]),
        .RESULT_DATA16(x2_res[16]), .RESULT_DATA17(x2_res[17]), .RESULT_DATA18(x2_res[18]), .RESULT_DATA19(x2_res[19]),
        .RESULT_DATA20(x2_res[20]), .RESULT_LAST(x2_res_last)
    );
endmodule
