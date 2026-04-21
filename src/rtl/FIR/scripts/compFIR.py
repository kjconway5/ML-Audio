#!/usr/bin/env python3

import numpy as np
import math
import argparse
import sys

#System parameters
FIN   = 1_008_000   # CIC input sample rate (Hz)
FOUT  =    16_000   # CIC output / FIR sample rate (Hz)
FNYQ  =  FOUT // 2  # FIR Nyquist (8 kHz)
R     =        63   # CIC decimation ratio
N_CIC =         3   # CIC stages

# Design parameters (also settable via CLI)
NTAPS  = 33         # FIR length (must be odd for Type I)
FPASS  = 7_000      # Passband edge (Hz)   — compensate fully up to here
BITS   = 14         # Coefficient word width (signed two's-complement)
KAISER_BETA = 6.0   # Kaiser window shape (6 → ~−44 dB sidelobes)
IW     = 16         # Input word width for generated Verilog
NFFT   = 4096       # Frequency grid for coefficient design


# CIC magnitude response
def cic_mag(f: np.ndarray) -> np.ndarray:
    """Normalised CIC magnitude |H(f)| for N stages, ratio R."""
    f = np.asarray(f, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        num = np.abs(np.sin(np.pi * f / FOUT))
        den = R * np.abs(np.sin(np.pi * f / FIN))
        h   = np.where(den < 1e-12, 1.0, (num / den) ** N_CIC)
    return h


# Kaiser window 
def kaiser_window(n: int, beta: float) -> np.ndarray:
    from numpy import i0
    k  = np.arange(n)
    x  = 2.0 * k / (n - 1) - 1.0
    return i0(beta * np.sqrt(np.maximum(1.0 - x**2, 0.0))) / i0(beta)


# FIR design (frequency-sampling + Kaiser window)
def design_fir(ntaps: int, fpass: float, bits: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (h_float, h_quantised) where:
      h_float     — ideal windowed coefficients, DC gain = 1
      h_quantised — signed integers, range ±(2^(bits-1)-1)
    """
    assert ntaps % 2 == 1, "NTAPS must be odd (Type I FIR)"
    M = (ntaps - 1) // 2

    # Build desired frequency response on NFFT-point grid (one-sided)
    freqs = np.linspace(0, FNYQ, NFFT // 2 + 1)
    cic   = cic_mag(freqs)
    D     = np.zeros(NFFT // 2 + 1)

    pb_mask  = freqs <= fpass
    trans_mask = (freqs > fpass)

    # Passband: exact inverse-CIC compensation
    D[pb_mask] = np.where(cic[pb_mask] > 1e-6, 1.0 / cic[pb_mask], 0.0)

    # Transition + stopband: cosine taper from 1/CIC → 0
    t  = (freqs[trans_mask] - fpass) / (FNYQ - fpass)
    wt = 0.5 * (1.0 + np.cos(np.pi * t))
    D[trans_mask] = np.where(cic[trans_mask] > 1e-6,
                             wt / cic[trans_mask], 0.0)

    # Reconstruct symmetric FIR via cosine-series IDFT
    ns = np.arange(NFFT // 2 + 1)
    h  = np.zeros(ntaps)
    for k in range(M + 1):
        phase = 2.0 * np.pi * k * ns / NFFT
        val   = D[0] + 2.0 * np.sum(D[1:-1] * np.cos(phase[1:-1])) \
                + D[-1] * np.cos(np.pi * k)
        val  /= NFFT
        h[M + k] = val
        h[M - k] = val

    # Kaiser window
    h *= kaiser_window(ntaps, KAISER_BETA)

    # Normalise to unity DC gain
    h /= h.sum()

    # Quantise to signed integer
    scale = (1 << (bits - 1)) - 1
    hq    = np.round(h * scale).astype(int)
    return h, hq


# CSD conversion (Non-Adjacent Form)
def to_csd(n: int) -> list[tuple[int, int]]:
    """
    Convert integer n to CSD (Canonical Signed Digit) representation.
    Returns list of (bit_position, sign) for each non-zero digit.
    The NAF guarantee: no two consecutive nonzero digits.
    """
    terms = []
    v = abs(n)
    bit = 0
    while v:
        if v & 1:
            d = -1 if (v & 3) == 3 else 1
            terms.append((bit, d))
            v -= d
        v >>= 1
        bit += 1
    if n < 0:
        terms = [(b, -s) for b, s in terms]
    return terms


def csd_str(terms: list[tuple[int, int]], val: int) -> str:
    """Human-readable CSD string e.g. '+2^13 -2^10 +2^7'."""
    if not terms:
        return "0"
    return "  ".join(f"{'+'if s>0 else '-'}2^{b}" for b, s in sorted(terms, reverse=True))


def csd_verilog_expr(sym_name: str, terms: list[tuple[int, int]], val: int,
                     iw: int, pw: int) -> str:
    """
    Generate a synthesisable Verilog shift-add expression for one CSD coefficient.
    sym_name — pre-adder result (signed, IW+1 bits wide)
    Returns  — combinational assign RHS (signed, PW bits)
    """
    if not terms or val == 0:
        return f"{pw}'sd0"
    ext   = f"$signed({{{pw - iw - 1}{{{{sym_name[{iw}]}}}}, {sym_name}}})"
    parts = []
    for b, s in sorted(terms, key=lambda x: -x[0]):
        shifted = f"({ext} <<< {b})" if b else ext
        parts.append((s, shifted))
    expr = parts[0][1] if parts[0][0] > 0 else f"(-{parts[0][1]})"
    for s, p in parts[1:]:
        expr += f" {'+'if s>0 else '-'} {p}"
    return expr


# Print design report
def print_report(h: np.ndarray, hq: np.ndarray, bits: int):
    M        = (len(h) - 1) // 2
    ntaps    = len(h)
    scale    = (1 << (bits - 1)) - 1
    freqs    = np.linspace(0, FNYQ, 2048)
    cic_r    = cic_mag(freqs)

    # FIR frequency response (Type I)
    fir_r = np.zeros(len(freqs))
    for i, f in enumerate(freqs):
        fnorm  = f / FOUT
        fir_r[i] = hq[M] / scale + 2.0 * sum(
            hq[M + k] / scale * np.cos(2 * np.pi * k * fnorm)
            for k in range(1, M + 1))
    comb_r = cic_r * fir_r

    # Stats
    pb_mask    = freqs <= FPASS
    cic_at7    = 20 * np.log10(max(cic_r[pb_mask][-1], 1e-12))
    fir_at7    = 20 * np.log10(max(abs(fir_r[pb_mask][-1]), 1e-12))
    comb_at7   = 20 * np.log10(max(abs(comb_r[pb_mask][-1]), 1e-12))
    pb_ripple  = np.ptp(20 * np.log10(np.abs(comb_r[pb_mask]) + 1e-12))

    print("=" * 70)
    print("  CIC Compensation FIR — Design Report")
    print("=" * 70)
    print(f"  FIR taps       : {ntaps}  (Type I symmetric, {M+1} unique)")
    print(f"  Passband edge  : {FPASS/1000:.1f} kHz")
    print(f"  Coeff bits     : {bits}")
    print(f"  Kaiser β       : {KAISER_BETA}")
    print(f"  CIC droop @7 kHz: {cic_at7:.2f} dB")
    print(f"  FIR boost @7 kHz: {fir_at7:.2f} dB")
    print(f"  Combined @7 kHz : {comb_at7:.2f} dB")
    print(f"  Passband ripple : ±{pb_ripple/2:.3f} dB")
    print()

    # CSD analysis per unique tap
    total_bin_adds = total_csd_adds = 0
    print(f"  {'Tap':<12}{'Float':>12}{'Fixed':>8}  {'CSD':<30}{'adds':>5}")
    print("  " + "-" * 67)
    for k in range(M + 1):
        idx   = M + k
        fv    = h[idx]
        qv    = hq[idx]
        mult  = 1 if k == 0 else 2
        csd   = to_csd(qv)
        bin_terms = max(0, int(abs(qv)).bit_length() - 1)
        csd_terms = len(csd)
        total_bin_adds += bin_terms * mult
        total_csd_adds += max(0, csd_terms - 1) * mult
        label = "center" if k == 0 else f"h[M±{k}]"
        print(f"  {label:<12}{fv:>12.7f}{qv:>8}  {csd_str(csd, qv):<30}{max(0,csd_terms-1):>5}")

    save = (1 - total_csd_adds / max(total_bin_adds, 1)) * 100
    print()
    print(f"  Binary adds total : {total_bin_adds}")
    print(f"  CSD adds total    : {total_csd_adds}")
    print(f"  Adder reduction   : {save:.1f}%")
    print("=" * 70)


# Verilog generation
VERILOG_HEADER = """\
////////////////////////////////////////////////////////////////////////////////
// ciccomp_fir.v — CIC Compensation FIR Filter (CSD arithmetic, ASIC)
// Auto-generated by design_cic_comp_fir.py
//
// Architecture:
//   * Type I symmetric FIR (linear phase, required for STFFT)
//   * Pre-adder structure  → halves the number of multiply operations
//   * CSD shift-add trees  → zero multiplier cells (hardwired shifts only)
//
// I/O widths:
//   IW  = input word width  (CIC output, truncated/rounded)
//   CW  = coefficient bits  ({bits})
//   OW  = output word width = IW + CW + ceil(log2(NTAPS)) + 1 guard
//
// Integration:
//   Connect CIC o_result (truncated to IW bits) → i_sample.
//   i_ce   = 1 every 16 kHz clock (one sample per cycle at 16 kHz, or
//            strobe if shared with a faster clock).
//   o_result feeds directly into STFFT i_sample port.
//
//   Synthesis tip:
//     set_dont_use [get_lib_cells *MULT*]
//   This enforces the multiplier-free netlist.
////////////////////////////////////////////////////////////////////////////////
`default_nettype none

module ciccomp_fir #(
    parameter NTAPS = {ntaps},   // Total taps (must be odd)
    parameter IW    = {iw},      // Input word width
    parameter CW    = {bits},    // Coefficient word width
    parameter OW    = {ow}       // Output width = IW+CW+ceil(log2(NTAPS))+1
) (
    input  wire               i_clk,
    input  wire               i_reset,
    input  wire               i_ce,
    input  wire signed [IW-1:0] i_sample,
    output reg  signed [OW-1:0] o_result
);

    localparam M  = (NTAPS-1)/2;   // Index of centre tap
    localparam PW = IW + CW + 1;   // Pre-adder product width (before accumulate)

"""

def generate_verilog(hq: np.ndarray, ntaps: int, bits: int, iw: int) -> str:
    M   = (ntaps - 1) // 2
    ow  = iw + bits + math.ceil(math.log2(ntaps)) + 1
    pw  = iw + bits + 1

    out = VERILOG_HEADER.format(ntaps=ntaps, iw=iw, bits=bits, ow=ow)

    # Delay line
    out += "    // ── Delay line ──────────────────────────────────────────────────────\n"
    out += f"    reg signed [IW-1:0] sr [0:NTAPS-1];\n"
    out += "    integer i;\n"
    out += "    always @(posedge i_clk)\n"
    out += "    if (i_reset) begin\n"
    out += "        for (i = 0; i < NTAPS; i = i+1) sr[i] <= 0;\n"
    out += "    end else if (i_ce) begin\n"
    out += "        for (i = NTAPS-1; i > 0; i = i-1) sr[i] <= sr[i-1];\n"
    out += "        sr[0] <= i_sample;\n"
    out += "    end\n\n"

    # Pre-adders
    out += "    // ── Pre-adders (exploit symmetry: sr[k] + sr[NTAPS-1-k]) ────────────\n"
    out += f"    wire signed [IW:0] sym [0:M];\n"
    out += "    genvar k;\n"
    out += "    generate\n"
    out += "    for (k = 0; k < M; k = k+1) begin : PREADD\n"
    out += "        assign sym[k] = $signed({sr[k][IW-1], sr[k]}) +\n"
    out += "                        $signed({sr[NTAPS-1-k][IW-1], sr[NTAPS-1-k]});\n"
    out += "    end\n"
    out += "    endgenerate\n"
    out += "    assign sym[M] = $signed({sr[M][IW-1], sr[M]});  // centre tap\n\n"

    # CSD products
    out += f"    // ── CSD multiply (shift-add, no multiplier cells) ───────────────────\n"
    out += f"    // PW = IW+CW+1 = {pw}\n"
    out += f"    wire signed [OW-1:0] prod [0:M];\n\n"

    for k in range(M + 1):
        idx  = M + k
        qv   = int(hq[idx])
        csd  = to_csd(qv)
        symn = f"sym[{k}]"
        # sign-extend sym (IW+1 bits) to PW bits
        # sym is already IW+1 bits; shift further extends as signed

        if not csd or qv == 0:
            out += f"    assign prod[{k}] = {ow}'sd0;  // h={hq[idx]}\n"
            continue

        terms_str_parts = []
        for b, s in sorted(csd, key=lambda x: -x[0]):
            # sign-extend (IW+1)-bit sym to OW bits, then shift
            # Verilog: $signed({{OW-IW-1{sym[k][IW]}}, sym[k]})
            ext = "$signed({{OW-IW-1{" + symn + "[IW]}}, " + symn + "})"
            if b == 0:
                shifted = ext
            else:
                shifted = "(" + ext + " <<< " + str(b) + ")"
            sign_str = "+" if s > 0 else "-"
            terms_str_parts.append((sign_str, shifted))

        # build expression
        expr_parts = []
        for i, (sg, sh) in enumerate(terms_str_parts):
            if i == 0:
                expr_parts.append(f"{'' if sg == '+' else '-'}{sh}")
            else:
                expr_parts.append(f" {sg} {sh}")

        expr  = "".join(expr_parts)
        cmt   = csd_str(csd, qv)
        out  += f"    // tap k={k}: {hq[idx]:+d}  CSD: {cmt}\n"
        out  += f"    assign prod[{k}] = {expr};\n\n"

    # Accumulator
    out += "    // ── Accumulator ──────────────────────────────────────────────────────\n"
    out += "    always @(posedge i_clk)\n"
    out += "    if (i_reset)\n"
    out += "        o_result <= 0;\n"
    out += "    else if (i_ce) begin\n"
    out += "        o_result <= prod[0]"
    for k in range(1, M + 1):
        out += f"\n                    + prod[{k}]"
    out += ";\n"
    out += "    end\n\n"
    out += "endmodule\n"
    return out


# Hex file for $readmemh 
def write_hex(hq: np.ndarray, bits: int, path: str):
    mask = (1 << bits) - 1
    with open(path, "w") as f:
        for v in hq:
            f.write(f"{int(v) & mask:0{(bits+3)//4}X}\n")
    print(f"  [hex] Coefficient hex file written → {path}")


# Main
def main():
    global NTAPS, FPASS, BITS, IW

    parser = argparse.ArgumentParser(description="CIC Compensation FIR Designer")
    parser.add_argument("--ntaps", type=int, default=NTAPS)
    parser.add_argument("--fpass", type=float, default=FPASS)
    parser.add_argument("--bits",  type=int, default=BITS)
    parser.add_argument("--iw",    type=int, default=IW,
                        help="Input word width for Verilog (default 16)")
    parser.add_argument("--out",   default="ciccomp_fir.v",
                        help="Output Verilog filename")
    parser.add_argument("--hex",   default="cic_comp_taps.hex",
                        help="Output hex filename (for $readmemh verification)")
    args = parser.parse_args()

    if args.ntaps % 2 == 0:
        print("Error: NTAPS must be odd (Type I FIR)"); sys.exit(1)

    NTAPS = args.ntaps
    FPASS = args.fpass
    BITS  = args.bits
    IW    = args.iw

    print(f"\nDesigning {NTAPS}-tap compensation FIR, passband {FPASS/1000:.1f} kHz, {BITS}-bit coefficients...\n")

    h, hq = design_fir(NTAPS, FPASS, BITS)
    print_report(h, hq, BITS)

    verilog = generate_verilog(hq, NTAPS, BITS, IW)
    with open(args.out, "w") as f:
        f.write(verilog)
    print(f"\n  [verilog] {args.out} written ({NTAPS} taps, {BITS}-bit CSD coefficients)")

    write_hex(hq, BITS, args.hex)
    print()


if __name__ == "__main__":
    main()