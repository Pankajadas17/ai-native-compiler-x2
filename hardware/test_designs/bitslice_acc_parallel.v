// Example eFPGA operator: shift-and-add accumulator for bit-sliced X2 weights.
// acc <= acc + (count << k) combines the 14-bit X2 counts of up to 8 weight bit-planes.
// All 22 accumulator bits come out in parallel: 14+3+3+22 = 42 pins > 36 IOBUF sites.
module bitslice_acc_parallel (
    input  wire        clk,
    input  wire        rst,
    input  wire        en,
    input  wire [13:0] count,
    input  wire [2:0]  k,
    output reg  [21:0] acc
);
    always @(posedge clk)
        if (rst)     acc <= 22'd0;
        else if (en) acc <= acc + ({8'd0, count} << k);
endmodule
