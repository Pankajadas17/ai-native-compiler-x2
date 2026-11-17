module sub(input [3:0] a, output [3:0] y); assign y = a; endmodule
module top(input [1:0] a, output [3:0] y);
  sub u0(.a(a), .y(y));
endmodule
