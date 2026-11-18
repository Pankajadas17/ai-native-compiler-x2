// SIMULATION ONLY (x2c): stand-in for the chip's RERAM_CIM_32x32 fabric primitive.
// Connects the primitive's ports to the BM Labs X2 behavioral model exactly as the chip does in
// sky130-reram-fpga src/fabric_wrapper.sv (fabric EXT_* pins l.259-312, macro instance l.566-624):
// MODE0/1 -> i_mode, ROW_INDEX0..4 -> i_row_index, ..., RESULT_DATA0..20 <- o_result_data,
// the 8 parameters (bitstream settings) -> the macro's configuration inputs. Scan/test pins tied to 0 as there.
// Every output passes through a 1 ns delay: the clock-to-output delay of the hard macro. Without it, fabric logic that
// samples on the rising edge races the behavioral model (which updates its outputs with blocking assignments at that
// same edge) and can miss a result in simulation. 1 ns << the 100 ns clock period used by the testbenches.
module RERAM_CIM_32x32 #(
    parameter [15:0] TARGET_SET1 = 0, parameter [15:0] TARGET_SET2 = 0,
    parameter [15:0] TARGET_RESET1 = 0, parameter [15:0] TARGET_RESET2 = 0,
    parameter [9:0]  NO_OF_CLK_CYCLES = 0, parameter [9:0] COUNTER_VALUE = 0,
    parameter [6:0]  TDC_TIME_OUT = 0, parameter [1:0] TDC_DEAD_TIME = 0
) (
    input  wire CLK, input wire RST, input wire CMD_VALID, output wire CMD_READY,
    input  wire MODE0, input wire MODE1,
    input  wire ROW_INDEX0, input wire ROW_INDEX1, input wire ROW_INDEX2, input wire ROW_INDEX3, input wire ROW_INDEX4,
    input  wire COL_INDEX0, input wire COL_INDEX1, input wire COL_INDEX2, input wire COL_INDEX3, input wire COL_INDEX4,
    input  wire PWM_DATA0, input wire PWM_DATA1, input wire PWM_DATA2, input wire PWM_DATA3,
    input  wire PWM_DATA4, input wire PWM_DATA5, input wire PWM_DATA6, input wire PWM_DATA7,
    input  wire COMPUTE_LAST, input wire COMPUTE_ALL_COLUMNS,
    output wire BUSY, output wire COMMAND_ACCEPTED, output wire COMPUTE_PENDING,
    output wire ERROR0, output wire ERROR1, output wire ERROR_VALID,
    output wire OPERATION_DONE, output wire PROGRAM_READBACK_ACTIVE,
    output wire RESULT_VALID, input wire RESULT_READY,
    output wire RESULT_DATA0,
    output wire RESULT_DATA1,
    output wire RESULT_DATA2,
    output wire RESULT_DATA3,
    output wire RESULT_DATA4,
    output wire RESULT_DATA5,
    output wire RESULT_DATA6,
    output wire RESULT_DATA7,
    output wire RESULT_DATA8,
    output wire RESULT_DATA9,
    output wire RESULT_DATA10,
    output wire RESULT_DATA11,
    output wire RESULT_DATA12,
    output wire RESULT_DATA13,
    output wire RESULT_DATA14,
    output wire RESULT_DATA15,
    output wire RESULT_DATA16,
    output wire RESULT_DATA17,
    output wire RESULT_DATA18,
    output wire RESULT_DATA19,
    output wire RESULT_DATA20,
    output wire RESULT_LAST
);
    wire [20:0] rd; wire [2:0] err;
    wire CMD_READY_m, BUSY_m, COMMAND_ACCEPTED_m, COMPUTE_PENDING_m, ERROR_VALID_m, OPERATION_DONE_m, PROGRAM_READBACK_ACTIVE_m, RESULT_VALID_m, RESULT_LAST_m;
    assign #1 CMD_READY = CMD_READY_m;
    assign #1 BUSY = BUSY_m;
    assign #1 COMMAND_ACCEPTED = COMMAND_ACCEPTED_m;
    assign #1 COMPUTE_PENDING = COMPUTE_PENDING_m;
    assign #1 ERROR_VALID = ERROR_VALID_m;
    assign #1 OPERATION_DONE = OPERATION_DONE_m;
    assign #1 PROGRAM_READBACK_ACTIVE = PROGRAM_READBACK_ACTIVE_m;
    assign #1 RESULT_VALID = RESULT_VALID_m;
    assign #1 RESULT_LAST = RESULT_LAST_m;
    assign #1 RESULT_DATA0 = rd[0];
    assign #1 RESULT_DATA1 = rd[1];
    assign #1 RESULT_DATA2 = rd[2];
    assign #1 RESULT_DATA3 = rd[3];
    assign #1 RESULT_DATA4 = rd[4];
    assign #1 RESULT_DATA5 = rd[5];
    assign #1 RESULT_DATA6 = rd[6];
    assign #1 RESULT_DATA7 = rd[7];
    assign #1 RESULT_DATA8 = rd[8];
    assign #1 RESULT_DATA9 = rd[9];
    assign #1 RESULT_DATA10 = rd[10];
    assign #1 RESULT_DATA11 = rd[11];
    assign #1 RESULT_DATA12 = rd[12];
    assign #1 RESULT_DATA13 = rd[13];
    assign #1 RESULT_DATA14 = rd[14];
    assign #1 RESULT_DATA15 = rd[15];
    assign #1 RESULT_DATA16 = rd[16];
    assign #1 RESULT_DATA17 = rd[17];
    assign #1 RESULT_DATA18 = rd[18];
    assign #1 RESULT_DATA19 = rd[19];
    assign #1 RESULT_DATA20 = rd[20];
    assign #1 ERROR0 = err[0]; assign #1 ERROR1 = err[1];   // the primitive exposes only bits 1:0 of the macro's 3-bit error
    sky130_bmlabs_reram__cim32x32 macro (
        .clk_i(CLK), .rst_i(RST),
        .i_cmd_valid(CMD_VALID), .o_cmd_ready(CMD_READY_m),
        .i_mode({MODE1, MODE0}),
        .i_row_index({ROW_INDEX4, ROW_INDEX3, ROW_INDEX2, ROW_INDEX1, ROW_INDEX0}),
        .i_col_index({COL_INDEX4, COL_INDEX3, COL_INDEX2, COL_INDEX1, COL_INDEX0}),
        .i_pwm_data({PWM_DATA7, PWM_DATA6, PWM_DATA5, PWM_DATA4, PWM_DATA3, PWM_DATA2, PWM_DATA1, PWM_DATA0}),
        .i_compute_last(COMPUTE_LAST), .i_compute_all_columns(COMPUTE_ALL_COLUMNS),
        .i_target_set1(TARGET_SET1), .i_target_set2(TARGET_SET2), .i_target_reset1(TARGET_RESET1), .i_target_reset2(TARGET_RESET2),
        .i_no_of_clk_cycles(NO_OF_CLK_CYCLES), .i_counter_value(COUNTER_VALUE),
        .i_tdc_time_out(TDC_TIME_OUT), .i_tdc_dead_time(TDC_DEAD_TIME),
        .o_busy(BUSY_m), .o_command_accepted(COMMAND_ACCEPTED_m), .o_compute_pending(COMPUTE_PENDING_m),
        .o_error(err), .o_error_valid(ERROR_VALID_m), .o_operation_done(OPERATION_DONE_m),
        .o_program_readback_active(PROGRAM_READBACK_ACTIVE_m),
        .o_result_valid(RESULT_VALID_m), .i_result_ready(RESULT_READY), .o_result_data(rd), .o_result_last(RESULT_LAST_m),
        .ScanInCC(1'b0), .ScanInDL(1'b0), .ScanInDR(1'b0), .ScanOutCC(), .TM(1'b0),
        .Iref(), .Vcc_read(), .Vcomp(), .Bias_comp2(), .Vcc_wl_read(), .Vcc_wl_set(), .Vbias(), .Vcc_wl_reset(), .Vcc_set(), .dc_bias()
    );
endmodule
