// SIMULATION ONLY (x2c): the IOBUF fabric primitive with the CHIP's enable polarity.
// The repo's sim model (ip/fabulous-tiles/primitives/IOBUF/yosys/primitives/IOBUF.v) drives the pad when EN = 0, but
//  * the techmap (primitives/IOBUF/yosys/techmap/IOBUF_map.v) ties EN = 1 on outputs and EN = 0 on inputs, and
//  * the tile (primitives/IOBUF/fabulous/IOBUF.v) passes EN out as EN_top -> fabric_io_oe_o (src/chip_core.sv l.54),
//    and src/double_wide_openframe_project_wrapper.v l.122 sets gpio_oeb = ~fabric_io_oe_o:
// on the chip EN = 1 drives the pin. With the repo model every output of a synthesized design floats in simulation.
// Same ports and parameters as the repo model; only the pad driver is changed.
module IOBUF #(
    parameter EN_REG = 1'b0,
    parameter IN_REG = 1'b0,
    parameter OUT_REG = 1'b0
) (
    input  CLK,
    input  EN,
    input  IN,
    output OUT,
    inout  PAD
);
    reg EN_q, IN_q, PAD_q;
    always @(posedge CLK) begin
        EN_q  <= EN;
        IN_q  <= IN;
        PAD_q <= PAD;
    end
    wire EN_mux  = EN_REG  ? EN_q  : EN;
    wire IN_mux  = IN_REG  ? IN_q  : IN;
    wire PAD_mux = OUT_REG ? PAD_q : PAD;
    assign PAD = EN_mux ? IN_mux : 1'bz;      // repo model: EN_mux ? 1'bz : IN_mux
    assign OUT = PAD_mux;
endmodule
