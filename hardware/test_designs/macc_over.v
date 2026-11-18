// Deliberately over capacity: 20 multipliers, the fabric has 18 MACC_8x8_20 blocks.
module macc_over (input wire clk, input wire [7:0] a, output wire q);
  reg [7:0] s [0:19];
  reg [15:0] p [0:19];
  integer i;
  reg x;
  always @(posedge clk) begin
    s[0] <= a;
    for (i = 1; i < 20; i = i + 1) s[i] <= s[i-1] ^ {s[i-1][6:0], s[i-1][7]};
    for (i = 0; i < 20; i = i + 1) p[i] <= s[i] * s[(i+7)%20];
    x = 1'b0;
    for (i = 0; i < 20; i = i + 1) x = x ^ (^p[i]);
  end
  reg qq; always @(posedge clk) qq <= x;
  assign q = qq;
endmodule
