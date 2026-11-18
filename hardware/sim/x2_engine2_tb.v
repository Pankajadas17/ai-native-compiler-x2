`timescale 1ns/1ps
// x2c testbench: a host driving hardware/designs/x2_engine2.v through its pins only, with ONE COMMAND QUEUE PER
// BLOCK running at the same time (so block 1 can be programmed while block 0 computes). The queues share the host port
// (cmd_data / strobes / acc_sel / byte_sel) through a lock; each waits on its own cmd_ready[b] and idle[b].
// Scripts (+SCRIPT0= +SCRIPT1=; an empty file = an idle block), one op per line:
//   "B mode row col pwm last all"  send a command to this queue's block (with the queue's current slice)
//   "P s"                           input slice of this block's next commands
//   "C"                             wait idle[b], clear bank b
//   "A n tile pos"                  wait idle[b], read accumulators 0..n-1 of bank b
// Log (+LOG=): "ACC <block> <tile> <pos> <index> <signed>", "DONE <commands> <reads> <cycles>".
module x2_engine2_tb;
  parameter integer CELL_BITS = 1;
  parameter [15:0] TARGET_SET1 = 0, TARGET_SET2 = 0, TARGET_RESET1 = 0, TARGET_RESET2 = 0;
  parameter [9:0]  NO_OF_CLK_CYCLES = 0, COUNTER_VALUE = 0;
  parameter [6:0]  TDC_TIME_OUT = 0;
  parameter [1:0]  TDC_DEAD_TIME = 0;
  reg clk = 0; always #50 clk = ~clk;
  integer cycles = 0; always @(posedge clk) cycles = cycles + 1;
  reg rst = 1;
  reg [11:0] cmd_data = 0; reg cmd_hi = 0, cmd_strobe = 0, acc_clear = 0;
  reg [3:0] acc_sel = 0; reg [1:0] byte_sel = 0;
  wire [1:0] cmd_ready, idle; wire [7:0] acc_byte;
  x2_engine2 #(.CELL_BITS(CELL_BITS), .TARGET_SET1(TARGET_SET1), .TARGET_SET2(TARGET_SET2), .TARGET_RESET1(TARGET_RESET1),
               .TARGET_RESET2(TARGET_RESET2), .NO_OF_CLK_CYCLES(NO_OF_CLK_CYCLES), .COUNTER_VALUE(COUNTER_VALUE),
               .TDC_TIME_OUT(TDC_TIME_OUT), .TDC_DEAD_TIME(TDC_DEAD_TIME))
    dut (.clk(clk), .rst(rst), .cmd_data(cmd_data), .cmd_hi(cmd_hi), .cmd_strobe(cmd_strobe), .cmd_ready(cmd_ready),
         .acc_clear(acc_clear), .idle(idle), .acc_sel(acc_sel), .byte_sel(byte_sel), .acc_byte(acc_byte));

  integer owner = -1;                         // host-port lock: -1 free, else the queue that holds it
  integer fout, cmds = 0, reads = 0, done_q = 0;
  reg [8*256-1:0] log_name;

  task automatic lock(input integer q);
    begin
      while (owner != -1) @(negedge clk);
      owner = q;                              // no suspension between the test and the claim: atomic in the simulator
    end
  endtask
  task automatic unlock; begin owner = -1; end endtask

  task automatic queue(input integer b, input [8*256-1:0] name);
    integer fin, code, quiet, k, n, m, r, c, p, l, a, tl, ps, sl;
    reg [8*8-1:0] tag; reg [21:0] word; reg [31:0] val;
    begin
      sl = 0;
      fin = $fopen(name, "r");
      while (fin != 0 && !$feof(fin)) begin
        code = $fscanf(fin, "%s", tag);
        if (code != 1) begin end
        else if (tag == "B") begin
          code = $fscanf(fin, "%d %d %d %d %d %d\n", m, r, c, p, l, a);
          word = {a[0], l[0], p[7:0], c[4:0], r[4:0], m[1:0]};
          while (cmd_ready[b] !== 1'b1) @(negedge clk);
          lock(b);
          cmd_data = word[11:0]; cmd_hi = 0; cmd_strobe = 1; @(negedge clk);
          cmd_data = {b[0], sl[0], word[21:12]}; cmd_hi = 1; @(negedge clk);
          cmd_strobe = 0;
          unlock();
          cmds = cmds + 1;
        end else if (tag == "P") begin
          code = $fscanf(fin, "%d\n", n); sl = n;
        end else if (tag == "C" || tag == "A") begin
          if (tag == "A") code = $fscanf(fin, "%d %d %d\n", n, tl, ps);
          quiet = 0;
          while (quiet < 8) begin @(negedge clk); quiet = (idle[b] === 1'b1) ? quiet + 1 : 0; end
          lock(b);
          if (tag == "C") begin
            acc_sel = {b[0], 3'd0}; acc_clear = 1; @(negedge clk); acc_clear = 0;
          end else begin
            for (k = 0; k < n; k = k + 1) begin
              acc_sel = {b[0], k[2:0]};
              byte_sel = 0; #1 val[7:0]   = acc_byte;
              byte_sel = 1; #1 val[15:8]  = acc_byte;
              byte_sel = 2; #1 val[23:16] = acc_byte;
              byte_sel = 3; #1 val[31:24] = acc_byte;
              $fdisplay(fout, "ACC %0d %0d %0d %0d %0d", b, tl, ps, k, $signed(val));
              reads = reads + 1;
            end
          end
          unlock();
        end
      end
      if (fin != 0) $fclose(fin);
      quiet = 0;
      while (quiet < 8) begin @(negedge clk); quiet = (idle[b] === 1'b1) ? quiet + 1 : 0; end
      done_q = done_q + 1;
    end
  endtask

  reg [8*256-1:0] s0, s1;
  initial begin
    if (!$value$plusargs("SCRIPT0=%s", s0) || !$value$plusargs("SCRIPT1=%s", s1) || !$value$plusargs("LOG=%s", log_name)) begin
      $display("need +SCRIPT0= +SCRIPT1= +LOG="); $finish;
    end
    fout = $fopen(log_name, "w");
    repeat (4) @(negedge clk); rst = 0; repeat (4) @(negedge clk);
    fork
      queue(0, s0);
      queue(1, s1);
    join
    $fdisplay(fout, "DONE %0d %0d %0d", cmds, reads, cycles);
    $fclose(fout); $finish;
  end
endmodule
