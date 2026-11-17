module relu8(input signed [7:0] x, output signed [7:0] y);
  assign y = x[7] ? 8'sd0 : x;
endmodule
