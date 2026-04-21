"""
diagnose_rtl.py  --  Compare RTL vs torchaudio numerically for specific frames.

Run:  python3 diagnose_rtl.py

Prints per-bin values for frames 5 and 20, and the per-frame mean/max of
the RTL output and torchaudio.  This pinpoints whether:
  A) Signal bins agree  → the problem is just noise floor (log clipping to 0)
  B) Signal bins differ → bfpexp magnitude is wrong or log LUT is miscalibrated
"""

import os, sys
import numpy as np
from pathlib import Path

_TOP_DIR  = os.path.dirname(os.path.abspath(__file__))
_ML_DIR   = os.path.normpath(os.path.join(_TOP_DIR, "..", "..", "ml"))
_PIPE_DIR = os.path.join(_ML_DIR, "Pipeline")
for p in (_ML_DIR, _PIPE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from features import LogMelExtractor

SAMPLE_RATE  = 16_000
SAMPLE_W     = 16
N_MELS       = 40
N_FFT        = 256
HOP          = N_FFT // 2       # 128 — matches RTL stfft
Q_FRAC       = 10               # RTL fixed-point fractional bits
STARTUP_LOSS = 2                # RTL pipeline startup frames to skip in golden
N_SAMPLES    = 7500


def make_chirp(n):
    dur   = n / SAMPLE_RATE
    t     = np.arange(n) / SAMPLE_RATE
    phase = 2 * np.pi * (200 * t + (7000 - 200) / (2 * dur) * t**2)
    return (np.sin(phase) * ((1 << (SAMPLE_W - 1)) - 1)).astype(np.int32)


def main():
    rtl_path = os.path.join(_TOP_DIR, "rtl_features.npy")
    if not os.path.exists(rtl_path):
        print("ERROR: rtl_features.npy not found. Run 'make test-cocotb' first.")
        return

    rtl_raw = np.load(rtl_path)                          # (40, n_frames) raw Q_FRAC ints
    rtl     = rtl_raw.astype(np.float32) / (1 << Q_FRAC) # convert to log2 float
    n_rtl   = rtl.shape[1]

    samples = make_chirp(N_SAMPLES)
    extractor = LogMelExtractor(
        sample_rate=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS,
        hop_length=HOP, window_length=N_FFT,
    )
    ta, _ = extractor.extract(samples)                    # (40, n_ta_frames) log2 float
    ta_aligned = ta[:, STARTUP_LOSS : STARTUP_LOSS + n_rtl]

    n_cmp = min(n_rtl, ta_aligned.shape[1])
    rtl_c = rtl[:, :n_cmp]
    ta_c  = ta_aligned[:, :n_cmp]

    print("=" * 65)
    print("RTL shape: %s    torchaudio aligned: %s" % (rtl.shape, ta_aligned.shape))
    print("=" * 65)

    # -----------------------------------------------------------------------
    # Per-frame statistics
    # -----------------------------------------------------------------------
    print("\nPer-frame: max RTL, max torchaudio, max |delta|")
    print("%-6s  %-10s  %-12s  %-10s" % ("Frame", "RTL max", "Torch max", "|Delta| max"))
    for f in range(n_cmp):
        rtl_max  = rtl_c[:, f].max()
        ta_max   = ta_c[:, f].max()
        delta    = abs(rtl_c[:, f] - ta_c[:, f]).max()
        # Only print frames where there's significant signal
        if ta_max > 5.0 or rtl_max > 5.0:
            print("%-6d  %-10.3f  %-12.3f  %-10.3f" % (f, rtl_max, ta_max, delta))

    # -----------------------------------------------------------------------
    # Detail: frame at 1/4 and 3/4 of the sequence
    # -----------------------------------------------------------------------
    for label, f in [("EARLY (frame 5)", 5), ("MID (frame %d)" % (n_cmp // 2), n_cmp // 2)]:
        if f >= n_cmp:
            continue
        print("\n%s" % label)
        print("%-8s  %-10s  %-12s  %-10s" % ("Mel bin", "RTL", "Torchaudio", "|Delta|"))
        for m in range(N_MELS):
            r = rtl_c[m, f]
            t = ta_c[m, f]
            d = abs(r - t)
            marker = " <-- signal" if (t > 10.0 or r > 10.0) else ""
            print("%-8d  %-10.3f  %-12.3f  %-10.3f%s" % (m, r, t, d, marker))

    # -----------------------------------------------------------------------
    # Key diagnosis
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("DIAGNOSIS")
    print("=" * 65)

    # Find which mel bin has the max torchaudio value in the middle frame
    mid = n_cmp // 2
    peak_bin_ta  = int(ta_c[:, mid].argmax())
    peak_bin_rtl = int(rtl_c[:, mid].argmax())
    rtl_at_ta_peak  = rtl_c[peak_bin_ta, mid]
    ta_at_ta_peak   = ta_c[peak_bin_ta, mid]

    print("Middle frame %d:" % mid)
    print("  torchaudio peak mel bin : %d  (value %.3f log2)" % (peak_bin_ta, ta_at_ta_peak))
    print("  RTL value at that bin   : %.3f log2" % rtl_at_ta_peak)
    print("  RTL peak mel bin        : %d  (value %.3f log2)" % (peak_bin_rtl, rtl_c[peak_bin_rtl, mid]))

    delta_at_peak = abs(rtl_at_ta_peak - ta_at_ta_peak)
    print("\n  Delta at signal bin     : %.3f log2" % delta_at_peak)

    # Count bins where RTL=0 but torchaudio > 1.0
    zero_mask = (rtl_c[:, mid] == 0) & (ta_c[:, mid] > 1.0)
    print("  Bins: RTL=0, torch>1.0  : %d / %d  (log floor clipping)" % (zero_mask.sum(), N_MELS))

    nonzero_rtl = rtl_c[:, mid][rtl_c[:, mid] > 0]
    if len(nonzero_rtl):
        print("  RTL non-zero bin values : min=%.3f  max=%.3f  mean=%.3f" % (
            nonzero_rtl.min(), nonzero_rtl.max(), nonzero_rtl.mean()))

    print("\nCONCLUSION GUIDE:")
    print("  If delta_at_peak < 2.0  --> signal bins match; issue is log floor only")
    print("                              Fix: add floor to log2_lut.hex entries near 0")
    print("  If delta_at_peak 2-10   --> small bfpexp scale error; adjust BFP_Q_FRAC")
    print("  If delta_at_peak > 10   --> bfpexp formula wrong, or log LUT miscalibrated")
    print("  If RTL peak_bin != torch peak_bin  --> bin alignment / HOP mismatch")


if __name__ == "__main__":
    main()