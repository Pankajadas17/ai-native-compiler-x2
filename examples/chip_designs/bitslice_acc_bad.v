// Deliberately broken (missing semicolon): used to check that synth failures are classified.
module bitslice_acc_bad (input wire clk, output reg q)
    always @(posedge clk) q <= ~q
endmodule
