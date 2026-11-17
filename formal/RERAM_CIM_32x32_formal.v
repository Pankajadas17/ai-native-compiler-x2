// FORMAL ONLY (bm_verif): an UNCONSTRAINED stand-in for the chip's X2 block (RERAM_CIM_32x32), for SymbiYosys.
// Instead of modelling what the X2 computes, it lets the solver choose every output in every clock, constrained only by
// the handshake rules the engine relies on (the BM behavioral model keeps them):
//   * no result while in reset;
//   * a result stays valid, with the same data, until RESULT_READY takes it;
//   * the block takes no command while it holds a result (x2_engine2.v depends on this for the input-slice bookkeeping).
// So the engine's properties are checked against EVERY X2 behaviour that keeps these rules, not one simulated run.
module RERAM_CIM_32x32 #(
    parameter [15:0] TARGET_SET1 = '0, parameter [15:0] TARGET_SET2 = '0,
    parameter [15:0] TARGET_RESET1 = '0, parameter [15:0] TARGET_RESET2 = '0,
    parameter [ 9:0] NO_OF_CLK_CYCLES = '0, parameter [ 9:0] COUNTER_VALUE = '0,
    parameter [ 6:0] TDC_TIME_OUT = '0, parameter [ 1:0] TDC_DEAD_TIME = '0
) (
    (* clkbuf_sink *) input wire CLK,
    input wire RST,
    input wire CMD_VALID,
    input wire MODE0,
    input wire MODE1,
    input wire ROW_INDEX0,
    input wire ROW_INDEX1,
    input wire ROW_INDEX2,
    input wire ROW_INDEX3,
    input wire ROW_INDEX4,
    input wire COL_INDEX0,
    input wire COL_INDEX1,
    input wire COL_INDEX2,
    input wire COL_INDEX3,
    input wire COL_INDEX4,
    input wire PWM_DATA0,
    input wire PWM_DATA1,
    input wire PWM_DATA2,
    input wire PWM_DATA3,
    input wire PWM_DATA4,
    input wire PWM_DATA5,
    input wire PWM_DATA6,
    input wire PWM_DATA7,
    input wire COMPUTE_LAST,
    input wire COMPUTE_ALL_COLUMNS,
    output wire CMD_READY,
    output wire BUSY,
    output wire COMMAND_ACCEPTED,
    output wire COMPUTE_PENDING,
    output wire ERROR0,
    output wire ERROR1,
    output wire ERROR_VALID,
    output wire OPERATION_DONE,
    output wire PROGRAM_READBACK_ACTIVE,
    output wire RESULT_VALID,
    input wire RESULT_READY,
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
    (* anyseq *) reg        f_ready;
    (* anyseq *) reg        f_valid;
    (* anyseq *) reg [20:0] f_data;
    assign CMD_READY    = f_ready;
    assign RESULT_VALID = f_valid;
    assign {RESULT_DATA20, RESULT_DATA19, RESULT_DATA18, RESULT_DATA17, RESULT_DATA16, RESULT_DATA15, RESULT_DATA14, RESULT_DATA13, RESULT_DATA12, RESULT_DATA11, RESULT_DATA10, RESULT_DATA9, RESULT_DATA8, RESULT_DATA7, RESULT_DATA6, RESULT_DATA5, RESULT_DATA4, RESULT_DATA3, RESULT_DATA2, RESULT_DATA1, RESULT_DATA0} = f_data;
    assign {BUSY, COMMAND_ACCEPTED, COMPUTE_PENDING, ERROR0, ERROR1, ERROR_VALID, OPERATION_DONE, PROGRAM_READBACK_ACTIVE, RESULT_LAST} = '0;

    reg f_past = 1'b0;
    always @(posedge CLK) f_past <= 1'b1;
    always @(*) begin
        if (RST) assume(!f_valid);
        if (f_valid) assume(!f_ready);
    end
    always @(posedge CLK) begin
        if (f_past && !$past(RST) && !RST && $past(f_valid) && !$past(RESULT_READY)) begin
            assume(f_valid);
            assume(f_data == $past(f_data));
        end
    end
endmodule
