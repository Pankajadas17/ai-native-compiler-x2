// x2c: the one-block X2 engine configured for 4-bit cells (8 accumulators, two 4-bit input slices).
// Built with x2_engine.v. Only its size and timing on the chip are checked: the chip's X2 commands can program
// 1-bit cells only, so 4-bit cells cannot be simulated on the chip's X2 model.
module x2_engine_4bit (
    input  wire clk, input wire rst,
    input  wire [11:0] cmd_data, input wire cmd_hi, input wire cmd_strobe, output wire cmd_ready,
    input  wire slice, input wire acc_clear, output wire idle,
    input  wire [2:0] acc_sel, input wire [1:0] byte_sel, output wire [7:0] acc_byte
);
    x2_engine #(.CELL_BITS(4)) engine (.clk(clk), .rst(rst), .cmd_data(cmd_data), .cmd_hi(cmd_hi), .cmd_strobe(cmd_strobe),
        .cmd_ready(cmd_ready), .slice(slice), .acc_clear(acc_clear), .idle(idle), .acc_sel(acc_sel), .byte_sel(byte_sel),
        .acc_byte(acc_byte));
endmodule
