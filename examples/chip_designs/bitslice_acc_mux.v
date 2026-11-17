// Same accumulator, output read one byte at a time (sel picks byte 0..2):
// 14+3+3+2+8 = 30 pins <= 36 IOBUF sites. Reading the result takes 3 transfers.
module bitslice_acc_mux (
    input  wire        clk,
    input  wire        rst,
    input  wire        en,
    input  wire [13:0] count,
    input  wire [2:0]  k,
    input  wire [1:0]  sel,
    output wire [7:0]  acc_byte
);
    reg [21:0] acc;
    always @(posedge clk)
        if (rst)     acc <= 22'd0;
        else if (en) acc <= acc + ({8'd0, count} << k);
    wire [23:0] acc_wide = {2'b00, acc};
    assign acc_byte = acc_wide[8*sel +: 8];
endmodule
