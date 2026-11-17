module requant_i32_i8(input signed [31:0] acc, output signed [7:0] y);
  wire signed [47:0] p = acc * 48'sd19 + 48'sd512;
  wire signed [47:0] q = p >>> 10;
  assign y = (q > 48'sd127) ? 8'sd127 : (q < -48'sd128) ? -8'sd128 : q[7:0];
endmodule
