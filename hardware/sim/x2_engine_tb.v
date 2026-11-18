`timescale 1ns/1ps
// x2c testbench: a host driving hardware/designs/x2_engine.v through its pins only.
// Script (+SCRIPT=): "CFG ..." (ignored; X2 settings come from -P), "B mode row col pwm last all" (send a command),
// "C" (clear accumulators), "P s" (input slice of the next pass), "W" (wait until idle for 8 clocks),
// "A n" (read accumulators 0..n-1, one byte at a time).  Log (+LOG=): "ACC <index> <signed>", "DONE <commands> <reads> <cycles>".
module x2_engine_tb;
  parameter integer CELL_BITS = 1;
  parameter [15:0] TARGET_SET1 = 0, TARGET_SET2 = 0, TARGET_RESET1 = 0, TARGET_RESET2 = 0;
  parameter [9:0]  NO_OF_CLK_CYCLES = 0, COUNTER_VALUE = 0;
  parameter [6:0]  TDC_TIME_OUT = 0;
  parameter [1:0]  TDC_DEAD_TIME = 0;
  reg clk = 0; always #50 clk = ~clk;
  integer cycles = 0; always @(posedge clk) cycles = cycles + 1;
  reg rst = 1;
  reg [11:0] cmd_data = 0; reg cmd_hi = 0, cmd_strobe = 0, slice = 0, acc_clear = 0;
  reg [2:0] acc_sel = 0; reg [1:0] byte_sel = 0;
  wire cmd_ready, idle; wire [7:0] acc_byte;
  x2_engine #(.CELL_BITS(CELL_BITS), .TARGET_SET1(TARGET_SET1), .TARGET_SET2(TARGET_SET2), .TARGET_RESET1(TARGET_RESET1),
              .TARGET_RESET2(TARGET_RESET2), .NO_OF_CLK_CYCLES(NO_OF_CLK_CYCLES), .COUNTER_VALUE(COUNTER_VALUE),
              .TDC_TIME_OUT(TDC_TIME_OUT), .TDC_DEAD_TIME(TDC_DEAD_TIME))
    dut (.clk(clk), .rst(rst), .cmd_data(cmd_data), .cmd_hi(cmd_hi), .cmd_strobe(cmd_strobe), .cmd_ready(cmd_ready),
         .slice(slice), .acc_clear(acc_clear), .idle(idle), .acc_sel(acc_sel), .byte_sel(byte_sel), .acc_byte(acc_byte));

  integer fin, fout, code, cmds, reads, k, n, quiet, m, r, c, p, l, a, v0, v1, v2, v3, v4, v5, v6, v7;
  reg [8*8-1:0] tag; reg [8*256-1:0] script_name, log_name; reg [21:0] word; reg [31:0] val;

  task send_half(input [11:0] d, input h);
    begin
      while (cmd_ready !== 1'b1) @(negedge clk);
      cmd_data = d; cmd_hi = h; cmd_strobe = 1; @(negedge clk); cmd_strobe = 0;
    end
  endtask

  task wait_idle;
    begin
      quiet = 0;
      while (quiet < 8) begin @(negedge clk); quiet = (idle === 1'b1) ? quiet + 1 : 0; end
    end
  endtask

  initial begin
    cmds = 0; reads = 0;
    if (!$value$plusargs("SCRIPT=%s", script_name) || !$value$plusargs("LOG=%s", log_name)) begin $display("need +SCRIPT= +LOG="); $finish; end
    fin = $fopen(script_name, "r"); fout = $fopen(log_name, "w");
    code = $fscanf(fin, "%s %d %d %d %d %d %d %d %d\n", tag, v0, v1, v2, v3, v4, v5, v6, v7);
    repeat (4) @(negedge clk); rst = 0; repeat (4) @(negedge clk);
    while (!$feof(fin)) begin
      code = $fscanf(fin, "%s", tag);
      if (code != 1) begin end
      else if (tag == "B") begin
        code = $fscanf(fin, "%d %d %d %d %d %d\n", m, r, c, p, l, a);
        word = {a[0], l[0], p[7:0], c[4:0], r[4:0], m[1:0]};
        send_half(word[11:0], 1'b0); send_half({2'b00, word[21:12]}, 1'b1);
        cmds = cmds + 1;
      end else if (tag == "C") begin
        wait_idle(); acc_clear = 1; @(negedge clk); acc_clear = 0;
      end else if (tag == "P") begin
        code = $fscanf(fin, "%d\n", n); wait_idle(); slice = n[0];
      end else if (tag == "W") begin
        wait_idle();
      end else if (tag == "A") begin
        code = $fscanf(fin, "%d\n", n);
        wait_idle();
        for (k = 0; k < n; k = k + 1) begin
          acc_sel = k;
          byte_sel = 0; #1 val[7:0]   = acc_byte;
          byte_sel = 1; #1 val[15:8]  = acc_byte;
          byte_sel = 2; #1 val[23:16] = acc_byte;
          byte_sel = 3; #1 val[31:24] = acc_byte;
          $fdisplay(fout, "ACC %0d %0d", k, $signed(val));
          reads = reads + 1;
        end
      end
    end
    wait_idle();
    $fdisplay(fout, "DONE %0d %0d %0d", cmds, reads, cycles);
    $fclose(fout); $finish;
  end
endmodule
