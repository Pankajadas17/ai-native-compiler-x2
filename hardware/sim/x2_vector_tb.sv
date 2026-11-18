`timescale 1ns/1ps
//------------------------------------------------------------------------------
// x2c vector testbench for the BM Labs X2 behavioral model
// (sky130_bmlabs_reram__cim32x32, read from the chip repo, never modified).
//
// Reads a vector file:
//   CFG <target_set1> <target_set2> <target_reset1> <target_reset2>
//       <no_of_clk_cycles> <counter_value> <tdc_time_out> <tdc_dead_time>
//   B <mode> <row> <col> <pwm> <compute_last> <compute_all_columns>    (one per beat)
// Writes a log:
//   R <mode> <col> <count> <last>     one line per result accepted
//   F <text>                          a protocol failure seen by the bench
//   DONE <beats> <results> <cycles>
//
// Handshake follows the vendor testbench: inputs change on negedge, the model
// samples on posedge, results are sampled on negedge with RESULT_READY held high.
//------------------------------------------------------------------------------
module x2_vector_tb;
  localparam integer MAX_WAIT = 20000;

  logic        clk_i = 1'b0;
  logic        rst_i = 1'b1;
  always #50 clk_i = ~clk_i;   // 10 MHz, as in the vendor bench

  logic        i_cmd_valid = 1'b0;
  wire         o_cmd_ready;
  logic [1:0]  i_mode = 2'b0;
  logic [4:0]  i_row_index = 5'b0;
  logic [4:0]  i_col_index = 5'b0;
  logic [7:0]  i_pwm_data = 8'b0;
  logic        i_compute_last = 1'b0;
  logic        i_compute_all_columns = 1'b0;
  logic [15:0] i_target_set1 = 16'b0, i_target_set2 = 16'b0;
  logic [15:0] i_target_reset1 = 16'b0, i_target_reset2 = 16'b0;
  logic [9:0]  i_no_of_clk_cycles = 10'b0, i_counter_value = 10'b0;
  logic [6:0]  i_tdc_time_out = 7'b0;
  logic [1:0]  i_tdc_dead_time = 2'b0;
  wire         o_busy, o_command_accepted, o_compute_pending, o_error_valid;
  wire [2:0]   o_error;
  wire         o_operation_done, o_program_readback_active;
  wire         o_result_valid;
  logic        i_result_ready = 1'b1;
  wire [20:0]  o_result_data;
  wire         o_result_last;
  logic        ScanInCC = 1'b0, ScanInDL = 1'b0, ScanInDR = 1'b0, TM = 1'b0;
  wire         ScanOutCC;

  sky130_bmlabs_reram__cim32x32 dut (
    .clk_i(clk_i), .rst_i(rst_i),
    .i_cmd_valid(i_cmd_valid), .o_cmd_ready(o_cmd_ready),
    .i_mode(i_mode), .i_row_index(i_row_index), .i_col_index(i_col_index),
    .i_pwm_data(i_pwm_data), .i_compute_last(i_compute_last),
    .i_compute_all_columns(i_compute_all_columns),
    .i_target_set1(i_target_set1), .i_target_set2(i_target_set2),
    .i_target_reset1(i_target_reset1), .i_target_reset2(i_target_reset2),
    .i_no_of_clk_cycles(i_no_of_clk_cycles), .i_counter_value(i_counter_value),
    .i_tdc_time_out(i_tdc_time_out), .i_tdc_dead_time(i_tdc_dead_time),
    .o_busy(o_busy), .o_command_accepted(o_command_accepted),
    .o_compute_pending(o_compute_pending), .o_error(o_error),
    .o_error_valid(o_error_valid), .o_operation_done(o_operation_done),
    .o_program_readback_active(o_program_readback_active),
    .o_result_valid(o_result_valid), .i_result_ready(i_result_ready),
    .o_result_data(o_result_data), .o_result_last(o_result_last),
    .ScanInCC(ScanInCC), .ScanInDL(ScanInDL), .ScanInDR(ScanInDR),
    .ScanOutCC(ScanOutCC), .TM(TM),
    .Iref(), .Vcc_read(), .Vcomp(), .Bias_comp2(), .Vcc_wl_read(),
    .Vcc_wl_set(), .Vbias(), .Vcc_wl_reset(), .Vcc_set(), .dc_bias()
  );

  integer fin, fout, code, beats, results, cycles, waited;
  integer v0, v1, v2, v3, v4, v5, v6, v7;
  integer m, r, c, p, l, a;
  reg [8*256-1:0] vec_name, log_name;
  reg [8*8-1:0]   tag;

  always @(posedge clk_i) cycles = cycles + 1;

  // Result monitor: valid lasts exactly one cycle per result while ready is high.
  always @(negedge clk_i) begin
    if (!rst_i && o_result_valid === 1'b1) begin
      $fdisplay(fout, "R %0d %0d %0d %0d", o_result_data[20:19],
                o_result_data[18:14], o_result_data[13:0], o_result_last);
      results = results + 1;
    end
  end

  task automatic wait_ready;
    begin
      waited = 0;
      while (o_cmd_ready !== 1'b1 && waited < MAX_WAIT) begin
        @(negedge clk_i);
        waited = waited + 1;
      end
      if (o_cmd_ready !== 1'b1) $fdisplay(fout, "F timeout waiting for cmd_ready before beat %0d", beats);
    end
  endtask

  task automatic wait_idle;
    begin
      waited = 0;
      while ((o_busy !== 1'b0 || o_result_valid !== 1'b0) && waited < MAX_WAIT) begin
        @(negedge clk_i);
        waited = waited + 1;
      end
      if (o_busy !== 1'b0) $fdisplay(fout, "F timeout waiting for idle at end");
      repeat (4) @(negedge clk_i);
    end
  endtask

  initial begin
    cycles = 0; beats = 0; results = 0;
    if (!$value$plusargs("VECTORS=%s", vec_name)) begin $display("missing +VECTORS="); $finish; end
    if (!$value$plusargs("LOG=%s", log_name)) begin $display("missing +LOG="); $finish; end
    fin  = $fopen(vec_name, "r");
    fout = $fopen(log_name, "w");
    if (fin == 0 || fout == 0) begin $display("cannot open vector or log file"); $finish; end

    code = $fscanf(fin, "%s %d %d %d %d %d %d %d %d\n", tag, v0, v1, v2, v3, v4, v5, v6, v7);
    if (code != 9 || tag != "CFG") begin $fdisplay(fout, "F bad CFG line"); $finish; end
    i_target_set1 = v0; i_target_set2 = v1; i_target_reset1 = v2; i_target_reset2 = v3;
    i_no_of_clk_cycles = v4; i_counter_value = v5; i_tdc_time_out = v6; i_tdc_dead_time = v7;

    repeat (4) @(negedge clk_i);
    rst_i = 1'b0;
    repeat (4) @(negedge clk_i);

    while (!$feof(fin)) begin
      code = $fscanf(fin, "%s %d %d %d %d %d %d\n", tag, m, r, c, p, l, a);
      if (code == 7 && tag == "B") begin
        wait_ready();
        @(negedge clk_i);
        i_mode = m; i_row_index = r; i_col_index = c; i_pwm_data = p;
        i_compute_last = l; i_compute_all_columns = a; i_cmd_valid = 1'b1;
        @(posedge clk_i);
        #1;
        if (o_command_accepted !== 1'b1) $fdisplay(fout, "F beat %0d not accepted", beats);
        @(negedge clk_i);
        i_cmd_valid = 1'b0; i_compute_last = 1'b0; i_compute_all_columns = 1'b0;
        beats = beats + 1;
      end else if (code > 0) begin
        $fdisplay(fout, "F bad vector line after beat %0d", beats);
      end
    end

    wait_idle();
    $fdisplay(fout, "DONE %0d %0d %0d", beats, results, cycles);
    $fclose(fout);
    $fclose(fin);
    $finish;
  end
endmodule
