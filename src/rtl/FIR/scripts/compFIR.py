#!/usr/bin/env python3
"""
design_cic_comp_fir.py — CIC Compensation FIR Designer

Generates compFIR.sv with:
  * AXI-Stream ready/valid handshake
  * Combinational next-state delay line (sr_next) so the pre-adders
    see the incoming sample on the same cycle it is accepted,
    giving correct 1-cycle latency
  * Pre-adders use: advance ? sr_next[k] : sr[k]
  * Type I symmetric CSD multiply, no multiplier cells

TRUNCATION RULE:
  The truncation in full_pipeline_top.sv must match the coefficient gain.
  This script prints the exact assign statement to use.
  Quick formula: shift = floor(log2(sum(full_h)))
"""

import numpy as np
import math
import argparse
import sys

# System parameters 
FIN          = 1_008_000
FOUT         =    16_000
FNYQ         = FOUT // 2
R            = FIN // FOUT
N_CIC        = 3

# Default design parameters
NTAPS        = 33
FPASS        = 7_000
BITS         = 14        # CW
KAISER_BETA  = 6.0
IW           = 16
NFFT         = 4096


# CIC magnitude
def cic_mag(f):
    f = np.asarray(f, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        num = np.abs(np.sin(np.pi * f / FOUT))
        den = R * np.abs(np.sin(np.pi * f / FIN))
        return np.where(den < 1e-12, 1.0, (num / den) ** N_CIC)


# Kaiser window
def kaiser_win(n, beta):
    from numpy import i0
    k = np.arange(n)
    x = 2.0 * k / (n - 1) - 1.0
    return i0(beta * np.sqrt(np.maximum(1.0 - x**2, 0.0))) / i0(beta)


# FIR design 
def design_fir(ntaps, fpass, bits):
    assert ntaps % 2 == 1, "NTAPS must be odd (Type I)"
    M     = (ntaps - 1) // 2
    freqs = np.linspace(0, FNYQ, NFFT // 2 + 1)
    cic   = cic_mag(freqs)
    D     = np.zeros(NFFT // 2 + 1)

    pb = freqs <= fpass
    tr = freqs > fpass
    D[pb] = np.where(cic[pb] > 1e-6, 1.0 / cic[pb], 0.0)
    t     = (freqs[tr] - fpass) / (FNYQ - fpass)
    D[tr] = np.where(cic[tr] > 1e-6, 0.5 * (1.0 + np.cos(np.pi * t)) / cic[tr], 0.0)

    ns = np.arange(NFFT // 2 + 1)
    h  = np.zeros(ntaps)
    for k in range(M + 1):
        phase = 2.0 * np.pi * k * ns / NFFT
        val   = (D[0]
                 + 2.0 * np.sum(D[1:-1] * np.cos(phase[1:-1]))
                 + D[-1] * np.cos(np.pi * k)) / NFFT
        h[M + k] = val
        h[M - k] = val

    h *= kaiser_win(ntaps, KAISER_BETA)
    h /= h.sum()  # normalise to unity DC gain in float

    max_c = (1 << (bits - 1)) - 1
    hq    = np.round(h * max_c).astype(int)
    return h, hq


# Effective hardware DC gain 
def eff_dc_gain(hq):
    """sum(full_h) = hardware DC gain (pre-adder doubles outer taps)."""
    M    = (len(hq) - 1) // 2
    half = hq[M:]                         # half[0]=outermost, half[M]=centre
    return 2 * int(sum(half[:-1])) + int(half[-1])


def safe_shift(hq, iw):
    """Return shift s such that (dc_out >> s) fits in signed iw-bit range."""
    g = eff_dc_gain(hq)
    x = (1 << (iw - 1)) - 1              # max positive input
    if g <= 0:
        return 0
    s = int(math.floor(math.log2(abs(g))))
    # Make sure no overflow even at peak passband gain (≈ +4 dB over DC)
    while x * abs(g) * 1.6 >= (1 << (iw - 1 + s)):
        s += 1
    return s


# CSD helpers
def to_csd(n):
    terms = []; v = abs(n); bit = 0
    while v:
        if v & 1:
            d = -1 if (v & 3) == 3 else 1
            terms.append((bit, d)); v -= d
        v >>= 1; bit += 1
    if n < 0:
        terms = [(b, -s) for b, s in terms]
    return terms


def csd_str(terms):
    if not terms:
        return "0"
    return "  ".join(
        f"{'+'if s>0 else '-'}2^{b}"
        for b, s in sorted(terms, reverse=True)
    )


# Design report
def print_report(hq, bits, iw):
    M    = (len(hq) - 1) // 2
    g    = eff_dc_gain(hq)
    s    = safe_shift(hq, iw)
    x    = (1 << (iw - 1)) - 1
    out  = (x * g) >> s
    gain_dB = 20 * np.log10(out / x + 1e-12)
    hi   = iw - 1 + s          # top bit of 16-bit slice
    lo   = s                   # bottom bit

    print("=" * 68)
    print("  CIC Compensation FIR — Design Report")
    print("=" * 68)
    print(f"  NTAPS         : {len(hq)}  (M = {M})")
    print(f"  Passband edge : {FPASS/1000:.1f} kHz  |  Kaiser beta = {KAISER_BETA}")
    print(f"  Coeff bits    : {bits}  (max = {(1<<(bits-1))-1})")
    print(f"  sum(full_h)   : {g}")
    print(f"  Shift         : {s}  → divide by 2^{s} = {2**s}")
    print(f"  DC gain check : input={x} → output={out}  ({gain_dB:+.2f} dB from unity)")
    print()
    print(f"  ┌─ UPDATE full_pipeline_top.sv ────────────────────────────────")
    print(f"  │  // Correct truncation for these coefficients:")
    print(f"  │  assign fir_trunc = fir_tdata[{hi}:{lo}];")
    print(f"  │  // Gain = {g}/2^{s} = {g/2**s:.4f}  ({gain_dB:+.2f} dB)")
    print(f"  └──────────────────────────────────────────────────────────────")
    print()
    print(f"  Half coefficients (k=0 outermost → k={M} centre):")
    print(f"  {'k':>4}  {'value':>8}  CSD")
    print("  " + "-" * 52)
    for k, v in enumerate(hq[M:]):
        print(f"  {k:>4}  {v:>8}  {csd_str(to_csd(v))}")
    print("=" * 68)


# Verilog generation
def gen_csd_expr(sym_name: str, qv: int, ow: int, iw: int) -> str:
    """CSD shift-add expression for one tap, OW-bit signed output."""
    csd = to_csd(qv)
    if not csd or qv == 0:
        return f"{ow}'sd0"
    ext = f"$signed({{{{OW-IW-1{{{sym_name}[IW]}}}}, {sym_name}}})"
    parts = sorted(csd, key=lambda x: -x[0])
    terms = [(s, f"({ext} <<< {b})" if b else ext) for b, s in parts]
    expr  = terms[0][1] if terms[0][0] > 0 else f"(-{terms[0][1]})"
    for s, p in terms[1:]:
        expr += f" {'+'if s>0 else '-'} {p}"
    return expr


def generate_verilog(hq, ntaps, bits, iw):
    M    = (ntaps - 1) // 2
    ow   = iw + bits + math.ceil(math.log2(ntaps)) + 1
    g    = eff_dc_gain(hq)
    s    = safe_shift(hq, iw)
    hi   = iw - 1 + s
    lo   = s
    HALF = hq[M:]   # HALF[0]=outermost tap, HALF[M]=centre tap

    L = []

    L.append("// Auto-generated by design_cic_comp_fir.py")
    L.append("/*")
    L.append("Architecture:")
    L.append("  * Type I symmetric FIR (linear phase, required for STFFT)")
    L.append("  * Pre-adder structure  -> halves the number of multiply operations")
    L.append("  * CSD shift-add trees  -> zero multiplier cells (hardwired shifts only)")
    L.append("  * AXI-Stream ready/valid handshake on both input and output ports")
    L.append("  * Combinational next-state delay line (sr_next) for correct 1-cycle latency")
    L.append("")
    L.append("I/O widths:")
    L.append(f"  IW  = input word width  (CIC output, truncated/rounded)")
    L.append(f"  CW  = coefficient bits  ({bits})")
    L.append(f"  OW  = output word width = IW + CW + ceil(log2(NTAPS)) + 1 guard")
    L.append("")
    L.append("TRUNCATION (update full_pipeline_top.sv):")
    L.append(f"  assign fir_trunc = fir_tdata[{hi}:{lo}];")
    L.append(f"  // sum(full_h)={g}, shift={s}, DC gain={g/2**s:.4f} ({20*np.log10(g/2**s+1e-12):+.2f} dB)")
    L.append("*/")
    L.append("")
    L.append("//     set_dont_use [get_lib_cells *MULT*]")
    L.append("")
    L.append("`default_nettype none")
    L.append("")
    L.append("module compFIR #(")
    L.append(f"    parameter NTAPS = {ntaps},   // Total taps (must be odd)")
    L.append(f"    parameter IW    = {iw},   // Input word width")
    L.append(f"    parameter CW    = {bits},    // Coefficient word width")
    L.append(f"    parameter OW    = {ow}    // Output width = IW+CW+ceil(log2(NTAPS))+1")
    L.append(") (")
    L.append("    input  wire                   i_clk,")
    L.append("    input  wire                   i_reset,")
    L.append("")
    L.append("    // Upstream (from CIC decimator)")
    L.append("    input  wire signed [IW-1:0]   i_tdata,")
    L.append("    input  wire                   i_tvalid,")
    L.append("    output wire                   i_tready,")
    L.append("")
    L.append("    // Downstream (to STFFT)")
    L.append("    output reg  signed [OW-1:0]   o_tdata,")
    L.append("    output reg                    o_tvalid,")
    L.append("    input  wire                   o_tready")
    L.append(");")
    L.append("")
    L.append("    localparam M = (NTAPS-1)/2;   // Index of centre tap")
    L.append("")
    L.append("    // Handshake: 1-deep pipeline, i_tready = !o_tvalid || o_tready")
    L.append("    assign i_tready = !o_tvalid || o_tready;")
    L.append("    wire advance = i_tvalid && i_tready;  // new sample accepted this cycle")
    L.append("")
    L.append("    // Registered delay line")
    L.append("    reg signed [IW-1:0] sr [0:NTAPS-1];")
    L.append("    integer i;")
    L.append("    always @(posedge i_clk) begin")
    L.append("        if (i_reset) begin")
    L.append("            for (i = 0; i < NTAPS; i = i+1) sr[i] <= 0;")
    L.append("        end else if (advance) begin")
    L.append("            for (i = NTAPS-1; i > 0; i = i-1) sr[i] <= sr[i-1];")
    L.append("            sr[0] <= i_tdata;")
    L.append("        end")
    L.append("    end")
    L.append("")
    L.append("    // Combinational next-state delay line")
    L.append("    //")
    L.append("    // sr_next[k] is the value sr[k] WILL hold after the next advance.")
    L.append("    // The pre-adders use (advance ? sr_next[k] : sr[k]) so the output")
    L.append("    // register captures the correct sum in the same cycle as advance,")
    L.append("    // giving exactly 1-cycle latency from input to output.")
    L.append("    wire signed [IW-1:0] sr_next [0:NTAPS-1];")
    L.append("    assign sr_next[0] = i_tdata;")
    L.append("    genvar si;")
    L.append("    generate")
    L.append("    for (si = 1; si < NTAPS; si = si+1) begin : SR_NEXT_GEN")
    L.append("        assign sr_next[si] = sr[si-1];")
    L.append("    end")
    L.append("    endgenerate")
    L.append("")
    L.append("    // Pre-adders  (exploit Type-I symmetry: tap[k] + tap[NTAPS-1-k])")
    L.append("    wire signed [IW:0] sym [0:M];")
    L.append("    genvar k;")
    L.append("    generate")
    L.append("    for (k = 0; k < M; k = k+1) begin : PREADD")
    L.append("        wire signed [IW-1:0] a = advance ? sr_next[k]          : sr[k];")
    L.append("        wire signed [IW-1:0] b = advance ? sr_next[NTAPS-1-k]  : sr[NTAPS-1-k];")
    L.append("        assign sym[k] = $signed({a[IW-1], a}) + $signed({b[IW-1], b});")
    L.append("    end")
    L.append("    endgenerate")
    L.append("    wire signed [IW-1:0] mid = advance ? sr_next[M] : sr[M];")
    L.append("    assign sym[M] = $signed({mid[IW-1], mid});")
    L.append("")
    L.append("    // CSD multiply  (shift-add, no multiplier cells)")
    L.append(f"    // PW = IW+CW+1 = {iw+bits+1}")
    L.append("    wire signed [OW-1:0] prod [0:M];")
    L.append("")

    for k in range(M + 1):
        qv   = int(HALF[k])
        symn = f"sym[{k}]"
        csd  = to_csd(qv)
        expr = gen_csd_expr(symn, qv, ow, iw)
        L.append(f"    // tap k={k}: {qv:+d}  CSD: {csd_str(csd)}")
        L.append(f"    assign prod[{k}] = {expr};")
        L.append("")

    L.append("    // Output register — advances only on accepted samples")
    L.append("    always @(posedge i_clk) begin")
    L.append("        if (i_reset) begin")
    L.append("            o_tdata  <= 0;")
    L.append("            o_tvalid <= 0;")
    L.append("        end else if (advance) begin")
    L.append("            o_tdata  <= prod[0]")
    for k in range(1, M + 1):
        L.append(f"                       + prod[{k}]")
    L.append("                       ;")
    L.append("            o_tvalid <= 1;")
    L.append("        end else if (o_tready) begin")
    L.append("            // Downstream consumed output but no new sample arrived.")
    L.append("            o_tvalid <= 0;")
    L.append("        end")
    L.append("    end")
    L.append("")
    L.append("endmodule")

    return "\n".join(L) + "\n"


# Main
def main():
    global NTAPS, FPASS, BITS, IW

    parser = argparse.ArgumentParser(
        description="CIC Compensation FIR Designer (ready/valid, next-state delay line)"
    )
    parser.add_argument("--ntaps", type=int,   default=NTAPS)
    parser.add_argument("--fpass", type=float, default=FPASS)
    parser.add_argument("--bits",  type=int,   default=BITS)
    parser.add_argument("--iw",    type=int,   default=IW)
    parser.add_argument("--out",   default="compFIR.sv")
    args = parser.parse_args()

    if args.ntaps % 2 == 0:
        print("Error: NTAPS must be odd (Type I FIR)")
        sys.exit(1)

    NTAPS = args.ntaps
    FPASS = args.fpass
    BITS  = args.bits
    IW    = args.iw

    print(f"\nDesigning {NTAPS}-tap CIC compensation FIR, "
          f"passband {FPASS/1000:.1f} kHz, {BITS}-bit coefficients ...\n")

    h, hq = design_fir(NTAPS, FPASS, BITS)
    print_report(hq, BITS, IW)

    verilog = generate_verilog(hq, NTAPS, BITS, IW)
    with open(args.out, "w") as f:
        f.write(verilog)
    print(f"\n[verilog] {args.out} written  ({NTAPS} taps, {BITS}-bit CSD, ready/valid + sr_next)")


if __name__ == "__main__":
    main()
