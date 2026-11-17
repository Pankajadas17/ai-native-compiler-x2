// FORMAL ONLY (bm_verif): properties of x2_engine2, included inside the module when FORMAL is defined (SymbiYosys).
// The chip flow and every simulation leave FORMAL undefined, so the engine they build is unchanged.
// Host contract (assumed): start in reset; clear a bank only while that bank is idle (the host waits for idle[b]).
// Proved (bounded, see formal/x2_engine2.sby):
//   P1 the arbiter never grants both blocks in the same clock;
//   P2 a pending command stays unchanged until its block takes it (no command lost or altered);
//   P3 whenever both blocks are idle, every accumulator equals a shadow model that adds each GRANTED result exactly
//      once (+/- count << shift), so no result is lost, doubled, or applied to the wrong bank or accumulator - this
//      covers the 3-stage pipeline and its forwarding.
    reg f_past = 1'b0;
    always @(posedge clk) f_past <= 1'b1;
    always @(*) if (!f_past) assume(rst);
    always @(*) if (acc_clear) assume(acc_sel[3] ? idle[1] : idle[0]);

    always @(*) if (!rst) assert(!(g0 && g1));                                           // P1

    always @(posedge clk) if (f_past && !$past(rst) && !rst) begin                       // P2
        if ($past(pend0) && !$past(x2_ready[0])) assert(pend0 && word0 == $past(word0) && slc_cmd0 == $past(slc_cmd0));
        if ($past(pend1) && !$past(x2_ready[1])) assert(pend1 && word1 == $past(word1) && slc_cmd1 == $past(slc_cmd1));
    end

    // P3: a shadow of ONE accumulator, chosen freely by the solver (so every accumulator is covered), updated once per
    // granted COMPUTE result for it, written independently of the pipeline.
    (* anyconst *) reg [3:0] f_k;
    always @(*) assume(f_k < 2 * NACC);
    reg  signed [31:0] f_sh;
    wire [20:0] f_res   = g1 ? rd1 : rd0;
    wire        f_slice = g1 ? slc_res1 : slc_res0;
    wire [4:0]  f_slot  = f_res[18:14] % CPW;
    wire [31:0] f_val   = {18'd0, f_res[13:0]} << ((f_slot % HALF) * CELL_BITS + (f_slice ? SPLIT : 0));
    wire [3:0]  f_idx   = (g1 ? NACC : 0) + f_res[18:14] / CPW;
    always @(posedge clk) begin
        if (rst || (f_k < NACC ? clr0 : clr1)) f_sh <= 32'sd0;
        else if ((g0 || g1) && f_res[20:19] == 2'd2 && f_idx == f_k)
            f_sh <= (f_slot >= HALF) ? f_sh - $signed(f_val) : f_sh + $signed(f_val);
    end
    always @(*) if (f_past && !rst && idle == 2'b11) assert(acc[f_k] == f_sh);                    // P3
