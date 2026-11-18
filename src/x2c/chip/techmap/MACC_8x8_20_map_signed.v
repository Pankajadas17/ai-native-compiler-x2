// x2c: corrected copy of fabulous-tiles primitives/MACC_8x8_20/yosys/techmap/MACC_8x8_20_map.v
// (mole99/fabulous-tiles @ 0651993f, 2026-11-10). Neither chip repo is edited: chip/chip_flow.py passes this file to the
// repo Makefile as  make MULTIPLIER_MAP_PATH="-multiplier-map <this file> 8 8 2 2 12"  (a command-line make variable
// overrides the value in user_designs/makefile.include l.26).
//
// What the original gets wrong:
//  1. It ignores A_SIGNED / B_SIGNED. The MACC_8x8_20 block multiplies UNSIGNED (tile RTL: product = OPA * OPB; SIGN_EXT
//     only widens that unsigned product to 20 bits). Yosys' mul2dsp splits a wide multiply into 8x8 pieces and marks the
//     pieces holding a sign bit signed, so every signed multiply wider than 8 bits came out wrong for negative operands.
//  2. It wires A[7..0] / B[7..0] even when the piece is narrower ("Range select out of bounds ... undef").
//
// The fix, with the block itself unchanged: an operand of width W with sign bit m is  value = unsigned - m * 2^W, so
//     a * b = a_u*b_u - am*(b_u << AW) - bm*(a_u << BW) + am*bm*2^(AW+BW)
// The block computes Q = a_u*b_u + C (unsigned, mod 2^20), so the correction terms go in through C; the block's own
// 20-bit adder adds them. Exact for Y_WIDTH <= 20 (mod 2^Y_WIDTH, as Yosys' $mul is). For unsigned pieces C = 0 as before.
// Verified exhaustively for every A_WIDTH, B_WIDTH in 1..8, both signednesses, several Y_WIDTH (tests/test_macc_signed_map.py).

module \$__FABULOUS_MUL (A, B, Y);
  parameter A_SIGNED = 0;
  parameter B_SIGNED = 0;
  parameter A_WIDTH = 0;
  parameter B_WIDTH = 0;
  parameter Y_WIDTH = 0;

	(* force_downto *)
	input [A_WIDTH-1:0] A;
	(* force_downto *)
	input [B_WIDTH-1:0] B;
	(* force_downto *)
	output [Y_WIDTH-1:0] Y;

	generate
	if (A_WIDTH > 8)
		wire _TECHMAP_FAIL_ = 1;
	else if (B_WIDTH > 8)
		wire _TECHMAP_FAIL_ = 1;
	else if (Y_WIDTH > 20)
		wire _TECHMAP_FAIL_ = 1;
	endgenerate

  // zero-extended 8-bit operands (fix 2: no out-of-range bits)
  wire [7:0] a8 = A;                        // A is unsigned here: assignment zero-extends
  wire [7:0] b8 = B;

  // sign bits that count as negative (fix 1)
  wire am = A_SIGNED ? A[A_WIDTH-1] : 1'b0;
  wire bm = B_SIGNED ? B[B_WIDTH-1] : 1'b0;

  wire [19:0] corr_a  = am ? ({12'd0, b8} << A_WIDTH) : 20'd0;
  wire [19:0] corr_b  = bm ? ({12'd0, a8} << B_WIDTH) : 20'd0;
  wire [19:0] corr_ab = (am & bm) ? (20'd1 << (A_WIDTH + B_WIDTH)) : 20'd0;

  // We need to combine individual bits into vectors
  // since FABulous doesn't support vectors for primitives yet
  wire [19:0] C;
  wire [19:0] Q;

  assign C = corr_ab - corr_a - corr_b;      // 0 for an unsigned piece (as in the original)
  assign Y = Q[Y_WIDTH-1:0];

  MACC_8x8_20 #(
    .A_REG    (1'b0),
    .B_REG    (1'b0),
    .C_REG    (1'b0),
    .SIGN_EXT (1'b0),
    .ACC_SEL  (1'b0),
    .ACC_OUT  (1'b0),
    .CLK_INV  (1'b0)
  ) _TECHMAP_REPLACE_ (
    .CLK  (),
    .EN   (1'b0),

    .ACC_RST (1'b0),
    .ACC_CI  (1'b0),
    .ACC_CO  (),

    .A0 (a8[0]),
    .A1 (a8[1]),
    .A2 (a8[2]),
    .A3 (a8[3]),
    .A4 (a8[4]),
    .A5 (a8[5]),
    .A6 (a8[6]),
    .A7 (a8[7]),

    .B0 (b8[0]),
    .B1 (b8[1]),
    .B2 (b8[2]),
    .B3 (b8[3]),
    .B4 (b8[4]),
    .B5 (b8[5]),
    .B6 (b8[6]),
    .B7 (b8[7]),

    .C0 (C[0]),
    .C1 (C[1]),
    .C2 (C[2]),
    .C3 (C[3]),
    .C4 (C[4]),
    .C5 (C[5]),
    .C6 (C[6]),
    .C7 (C[7]),
    .C8 (C[8]),
    .C9 (C[9]),
    .C10 (C[10]),
    .C11 (C[11]),
    .C12 (C[12]),
    .C13 (C[13]),
    .C14 (C[14]),
    .C15 (C[15]),
    .C16 (C[16]),
    .C17 (C[17]),
    .C18 (C[18]),
    .C19 (C[19]),

    .Q0 (Q[0]),
    .Q1 (Q[1]),
    .Q2 (Q[2]),
    .Q3 (Q[3]),
    .Q4 (Q[4]),
    .Q5 (Q[5]),
    .Q6 (Q[6]),
    .Q7 (Q[7]),
    .Q8 (Q[8]),
    .Q9 (Q[9]),
    .Q10 (Q[10]),
    .Q11 (Q[11]),
    .Q12 (Q[12]),
    .Q13 (Q[13]),
    .Q14 (Q[14]),
    .Q15 (Q[15]),
    .Q16 (Q[16]),
    .Q17 (Q[17]),
    .Q18 (Q[18]),
    .Q19 (Q[19])
  );

endmodule
