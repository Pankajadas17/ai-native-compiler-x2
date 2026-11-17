module add8_sat(input signed [7:0] a, input signed [7:0] b, output signed [7:0] y);
  wire signed [8:0] s = a + b;
  assign y = (s > 9'sd127) ? 8'sd127 : (s < -9'sd128) ? -8'sd128 : s[7:0];
endmodule
