// Deliberately broken: instantiates a module that does not exist.
module missing_mod (input wire a, output wire y);
  not_a_real_cell u0 (.a(a), .y(y));
endmodule
