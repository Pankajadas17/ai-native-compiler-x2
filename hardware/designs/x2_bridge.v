// x2c: X2 controller (host bridge) on the fabric.
// The chip's fabric memory (36 x RAM_32x4 = 4608 bits) cannot hold the weights, so the command stream made by
// compiler/x2_pack.py comes from a host through the pins, and this bridge hands it to the X2 block (RERAM_CIM_32x32,
// the chip's own primitive) and returns its results.
//
// Host side (30 of 36 pins):
//   command word (22 bits) = {all_cols, last, pwm[7:0], col[4:0], row[4:0], mode[1:0]}, loaded as two 12-bit halves:
//     cmd_hi=0 + cmd_strobe : store bits [11:0]
//     cmd_hi=1 + cmd_strobe : store bits [21:12] and send the command to the X2
//   cmd_ready  : 1 when a new command may be loaded (the previous one was accepted by the X2)
//   res_valid  : the X2 has a result; res_sel picks byte 0..2 of the 21-bit result {mode, column, count}; res_last
//   res_pop    : 1 for one clock after reading = the result is consumed (drives the X2's RESULT_READY)
// The X2 parameters are bitstream settings (parameters below). TDC_TIME_OUT must be at least what
// compiler/x2_pack.py reports for the layer (28 covers any full 32-row tile with 4-bit cells and 4-bit inputs).
module x2_bridge #(
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
    input  wire [1:0]  res_sel,
    input  wire        res_pop,
    output wire        res_valid,
    output wire        res_last,
    output wire [7:0]  res_byte
);
    reg [11:0] lo;
    reg [9:0]  hi;
    reg        pending;
    wire x2_cmd_ready, x2_res_valid, x2_res_last;
    wire [20:0] x2_res;
    wire [21:0] word = {hi, lo};

    always @(posedge clk) begin
        if (rst) begin
            pending <= 1'b0;
        end else begin
            if (cmd_strobe && !cmd_hi && !pending) lo <= cmd_data;
            if (cmd_strobe &&  cmd_hi && !pending) begin hi <= cmd_data[9:0]; pending <= 1'b1; end
            else if (pending && x2_cmd_ready)       pending <= 1'b0;      // accepted by the X2 on this edge
        end
    end

    assign cmd_ready = !pending;
    assign res_valid = x2_res_valid;
    assign res_last  = x2_res_last;
    assign res_byte  = (res_sel == 2'd0) ? x2_res[7:0] : (res_sel == 2'd1) ? x2_res[15:8] : {3'b000, x2_res[20:16]};

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
        .RESULT_VALID(x2_res_valid), .RESULT_READY(res_pop & x2_res_valid),
        .RESULT_DATA0(x2_res[0]), .RESULT_DATA1(x2_res[1]), .RESULT_DATA2(x2_res[2]), .RESULT_DATA3(x2_res[3]),
        .RESULT_DATA4(x2_res[4]), .RESULT_DATA5(x2_res[5]), .RESULT_DATA6(x2_res[6]), .RESULT_DATA7(x2_res[7]),
        .RESULT_DATA8(x2_res[8]), .RESULT_DATA9(x2_res[9]), .RESULT_DATA10(x2_res[10]), .RESULT_DATA11(x2_res[11]),
        .RESULT_DATA12(x2_res[12]), .RESULT_DATA13(x2_res[13]), .RESULT_DATA14(x2_res[14]), .RESULT_DATA15(x2_res[15]),
        .RESULT_DATA16(x2_res[16]), .RESULT_DATA17(x2_res[17]), .RESULT_DATA18(x2_res[18]), .RESULT_DATA19(x2_res[19]),
        .RESULT_DATA20(x2_res[20]), .RESULT_LAST(x2_res_last)
    );
endmodule
