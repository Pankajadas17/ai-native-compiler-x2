`timescale 1ns/1ps
// bm_verif testbench: a host driving examples/chip_designs/x2_bridge.v through its pins only.
// Reads the same vector file as sim/x2_vector_tb.sv ("CFG ..." line, then "B mode row col pwm last all").
// The X2 parameters are the bridge's parameters (set with iverilog -P). Log: "R mode col count last", "DONE beats results cycles".
module x2_bridge_tb;
  parameter [15:0] TARGET_SET1 = 0, TARGET_SET2 = 0, TARGET_RESET1 = 0, TARGET_RESET2 = 0;
  parameter [9:0]  NO_OF_CLK_CYCLES = 0, COUNTER_VALUE = 0;
  parameter [6:0]  TDC_TIME_OUT = 0;
  parameter [1:0]  TDC_DEAD_TIME = 0;
  reg clk = 0; always #50 clk = ~clk;
  reg rst = 1;
  reg [11:0] cmd_data = 0; reg cmd_hi = 0, cmd_strobe = 0, res_pop = 0; reg [1:0] res_sel = 0;
  wire cmd_ready, res_valid, res_last; wire [7:0] res_byte;
  x2_bridge #(.TARGET_SET1(TARGET_SET1), .TARGET_SET2(TARGET_SET2), .TARGET_RESET1(TARGET_RESET1), .TARGET_RESET2(TARGET_RESET2),
              .NO_OF_CLK_CYCLES(NO_OF_CLK_CYCLES), .COUNTER_VALUE(COUNTER_VALUE), .TDC_TIME_OUT(TDC_TIME_OUT),
              .TDC_DEAD_TIME(TDC_DEAD_TIME))
    dut (.clk(clk), .rst(rst), .cmd_data(cmd_data), .cmd_hi(cmd_hi), .cmd_strobe(cmd_strobe), .cmd_ready(cmd_ready),
         .res_sel(res_sel), .res_pop(res_pop), .res_valid(res_valid), .res_last(res_last), .res_byte(res_byte));

  integer cycles = 0; always @(posedge clk) cycles = cycles + 1;
  integer fin, fout, code, beats, results, quiet, m, r, c, p, l, a, v0, v1, v2, v3, v4, v5, v6, v7;
  reg [8*8-1:0] tag; reg [8*256-1:0] vec_name, log_name; reg [21:0] word; reg [20:0] res; reg done_sending;

  // host: read every result through the 8-bit port, then pop it
  always @(negedge clk) begin
    if (!rst && res_valid && !res_pop) begin
      res_sel = 0; #1 res[7:0]   = res_byte;
      res_sel = 1; #1 res[15:8]  = res_byte;
      res_sel = 2; #1 res[20:16] = res_byte[4:0];
      $fdisplay(fout, "R %0d %0d %0d %0d", res[20:19], res[18:14], res[13:0], res_last);
      results = results + 1;
      res_pop = 1;
    end else begin
      res_pop = 0;
    end
  end

  task send_half(input [11:0] d, input h);
    begin
      while (cmd_ready !== 1'b1) @(negedge clk);
      cmd_data = d; cmd_hi = h; cmd_strobe = 1; @(negedge clk); cmd_strobe = 0;
    end
  endtask

  initial begin
    beats = 0; results = 0; done_sending = 0;
    if (!$value$plusargs("VECTORS=%s", vec_name) || !$value$plusargs("LOG=%s", log_name)) begin $display("need +VECTORS= +LOG="); $finish; end
    fin = $fopen(vec_name, "r"); fout = $fopen(log_name, "w");
    code = $fscanf(fin, "%s %d %d %d %d %d %d %d %d\n", tag, v0, v1, v2, v3, v4, v5, v6, v7);   // CFG line: parameters come from -P
    repeat (4) @(negedge clk); rst = 0; repeat (4) @(negedge clk);
    while (!$feof(fin)) begin
      code = $fscanf(fin, "%s %d %d %d %d %d %d\n", tag, m, r, c, p, l, a);
      if (code == 7 && tag == "B") begin
        word = {a[0], l[0], p[7:0], c[4:0], r[4:0], m[1:0]};
        send_half(word[11:0], 1'b0);
        send_half({2'b00, word[21:12]}, 1'b1);
        beats = beats + 1;
      end
    end
    quiet = 0;                                   // finish after 2000 clocks without a result or a pending command
    while (quiet < 2000) begin @(negedge clk); quiet = (res_valid || !cmd_ready) ? 0 : quiet + 1; end
    $fdisplay(fout, "DONE %0d %0d %0d", beats, results, cycles);
    $fclose(fout); $finish;
  end
endmodule
