// x2c: the two-block X2 engine for 4-bit cells (hardware/designs/x2_engine2.v with CELL_BITS = 4).
module x2_engine2_4bit (
    input  wire clk, input wire rst,
    input  wire [11:0] cmd_data, input wire cmd_hi, input wire cmd_strobe, output wire [1:0] cmd_ready,
    input  wire acc_clear, output wire [1:0] idle,
    input  wire [3:0] acc_sel, input wire [1:0] byte_sel, output wire [7:0] acc_byte
);
    x2_engine2 #(.CELL_BITS(4)) engine (.clk(clk), .rst(rst), .cmd_data(cmd_data), .cmd_hi(cmd_hi), .cmd_strobe(cmd_strobe),
        .cmd_ready(cmd_ready), .acc_clear(acc_clear), .idle(idle), .acc_sel(acc_sel), .byte_sel(byte_sel), .acc_byte(acc_byte));
endmodule
