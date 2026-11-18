// x2c: fabric recombiner for X2 results.
// Rebuilds one layer output from X2 column counts: acc += (neg ? -1 : +1) * (count << shift), one step per clock.
// shift = input-slice scale + weight-digit scale (4-bit cells: 0, 4, 8, 12); neg = negative weight bank.
// The 32-bit result is read one byte at a time (sel), so the whole design needs 32 of the chip's 36 I/O pins.
// x2_acc_step is the operator checked by verification/rtl_check.py (A2 reference design).
module x2_recombiner (
    input  wire        clk,
    input  wire        clear,
    input  wire        en,
    input  wire        neg,
    input  wire [3:0]  shift,
    input  wire [13:0] count,
    input  wire [1:0]  sel,
    output wire [7:0]  acc_byte
);
    reg  signed [31:0] acc;
    wire signed [31:0] nxt;
    x2_acc_step step (.acc_in(acc), .count(count), .shift(shift), .neg(neg), .acc_out(nxt));
    always @(posedge clk)
        if (clear)   acc <= 32'sd0;
        else if (en) acc <= nxt;
    assign acc_byte = acc[8*sel +: 8];
endmodule

module x2_acc_step(input signed [31:0] acc_in, input [13:0] count, input [3:0] shift, input neg,
                   output signed [31:0] acc_out);
  wire [31:0] v = {18'd0, count} << shift;
  assign acc_out = neg ? acc_in - $signed(v) : acc_in + $signed(v);
endmodule
