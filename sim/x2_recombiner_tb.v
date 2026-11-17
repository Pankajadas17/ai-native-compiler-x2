`timescale 1ns/1ps
// bm_verif testbench for examples/chip_designs/x2_recombiner.v.
// Stimulus file (+STIM=): "C" = clear, "S <count> <shift> <neg>" = one step, "R" = read the 32-bit result
// one byte at a time through sel/acc_byte (as the chip pins would) and log it.  Log (+LOG=): "ACC <signed>", "DONE <n>".
module x2_recombiner_tb;
  reg clk = 0; always #5 clk = ~clk;
  reg clear = 0, en = 0, neg = 0;
  reg [3:0] shift = 0; reg [13:0] count = 0; reg [1:0] sel = 0;
  wire [7:0] acc_byte;
  x2_recombiner dut (.clk(clk), .clear(clear), .en(en), .neg(neg), .shift(shift), .count(count), .sel(sel), .acc_byte(acc_byte));
  integer fin, fout, code, c, s, n, reads;
  reg [8*8-1:0] tag;
  reg [31:0] word;
  reg [8*256-1:0] stim_name, log_name;
  initial begin
    reads = 0;
    if (!$value$plusargs("STIM=%s", stim_name) || !$value$plusargs("LOG=%s", log_name)) begin $display("need +STIM= +LOG="); $finish; end
    fin = $fopen(stim_name, "r"); fout = $fopen(log_name, "w");
    @(negedge clk);
    while (!$feof(fin)) begin
      code = $fscanf(fin, "%s", tag);
      if (code == 1 && tag == "C") begin
        clear = 1; @(negedge clk); clear = 0;
      end else if (code == 1 && tag == "S") begin
        code = $fscanf(fin, "%d %d %d\n", c, s, n);
        count = c; shift = s; neg = n; en = 1; @(negedge clk); en = 0;
      end else if (code == 1 && tag == "R") begin
        sel = 0; #1 word[7:0]   = acc_byte;
        sel = 1; #1 word[15:8]  = acc_byte;
        sel = 2; #1 word[23:16] = acc_byte;
        sel = 3; #1 word[31:24] = acc_byte;
        $fdisplay(fout, "ACC %0d", $signed(word));
        reads = reads + 1;
      end
    end
    $fdisplay(fout, "DONE %0d", reads);
    $fclose(fout); $finish;
  end
endmodule
