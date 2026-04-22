"""Verify that GoldenLegacyExtractor matches the legacy RTL pipeline output.

Loads the rtl_*_legacy.npy files (saved from the pre-fix pipeline_top run)
and compares against GoldenLegacyExtractor's output on the same input.
"""
import os, sys
import numpy as np

_TOP = os.path.dirname(os.path.abspath(__file__))
_ML  = os.path.normpath(os.path.join(_TOP, "..", "..", "ml"))
sys.path.insert(0, _ML)

from golden_model_legacy import GoldenLegacyExtractor
from golden_model import SAMPLE_RATE, SAMPLE_W, N_MELS, Q_FRAC

SAMPLE_MAX = (1 << (SAMPLE_W - 1)) - 1
N_SAMPLES = 7500


def make_chirp(n):
    dur = n / SAMPLE_RATE
    t = np.arange(n) / SAMPLE_RATE
    phase = 2 * np.pi * (200 * t + (7000 - 200) / (2 * dur) * t**2)
    return (np.sin(phase) * SAMPLE_MAX).astype(np.int32)


def main():
    rtl_re   = np.load(os.path.join(_TOP, "rtl_fft_re_legacy.npy"))
    rtl_im   = np.load(os.path.join(_TOP, "rtl_fft_im_legacy.npy"))
    rtl_bfp  = np.load(os.path.join(_TOP, "rtl_bfpexps_legacy.npy"))
    rtl_pre  = np.load(os.path.join(_TOP, "rtl_features_precomp_legacy.npy"))
    rtl_post = np.load(os.path.join(_TOP, "rtl_features_legacy.npy"))
    sync     = np.load(os.path.join(_TOP, "rtl_sync_sample_counts_legacy.npy"))

    Nrtl = rtl_re.shape[0]
    print(f"Legacy RTL has {Nrtl} frames, first 10 sync deltas = "
          f"{list(np.diff(sync[:11]))}")

    samples = make_chirp(N_SAMPLES)

    ext = GoldenLegacyExtractor()
    feats, gld_re, gld_im, gld_bfp = ext.extract_with_fft(samples)
    Nmod = gld_re.shape[0]
    print(f"GoldenLegacy emitted {Nmod} frames.")

    n = min(Nrtl, Nmod)
    print(f"\nComparing first {n} frames.\n")

    # --- Bit-exact FFT bin comparison ---
    re_eq = (gld_re[:n] == rtl_re[:n]).sum()
    im_eq = (gld_im[:n] == rtl_im[:n]).sum()
    total = n * 129
    print(f"[FFT bins bit-exact]  re: {re_eq}/{total} ({re_eq/total*100:.2f}%)   "
          f"im: {im_eq}/{total} ({im_eq/total*100:.2f}%)")

    # Peak-bin comparison
    rtl_mag = np.sqrt(rtl_re[:n].astype(np.float64)**2 + rtl_im[:n].astype(np.float64)**2)
    gld_mag = np.sqrt(gld_re[:n].astype(np.float64)**2 + gld_im[:n].astype(np.float64)**2)
    pk_match = sum(int(np.argmax(rtl_mag[k])) == int(np.argmax(gld_mag[k])) for k in range(1, n))
    print(f"[FFT peak bin match (excl. f0 startup)]  {pk_match}/{n-1}")

    # Bfpexp comparison (skip frame 0 which is the known-anomalous startup)
    bfp_eq = (gld_bfp[1:n] == rtl_bfp[1:n]).sum()
    print(f"[bfpexp match (excl. f0)]  {bfp_eq}/{n-1}")
    if bfp_eq < n - 1:
        diffs = [(k, int(rtl_bfp[k]), int(gld_bfp[k])) for k in range(1, n)
                 if gld_bfp[k] != rtl_bfp[k]]
        print(f"  mismatches: {diffs[:10]}")

    # --- Post-comp feature comparison ---
    rtl_post_f = rtl_post.astype(np.float32)   # already divided by 2^Q_FRAC
    gld_post_f = feats[:, :n].astype(np.float32) / (1 << Q_FRAC)
    rtl_post_cmp = rtl_post_f[:, :n]
    d = np.abs(gld_post_f - rtl_post_cmp)
    print(f"\n[POST-comp log features]  max|Δ|={d.max():.4f}  "
          f"mean|Δ|={d.mean():.4f}  median|Δ|={np.median(d):.4f}")

    # Worst frame breakdown
    wf = int(d.max(axis=0).argmax())
    print(f"  Worst frame f{wf}: max|Δ|={d[:,wf].max():.4f}  "
          f"bfp(R={int(rtl_bfp[wf])}/G={int(gld_bfp[wf])})")

    # Pre-comp comparison (no bfpexp contribution)
    rtl_pre_cmp = rtl_pre[:, :n]
    gld_pre_f = (feats[:, :n].astype(np.int32) -
                 (2 * gld_bfp[:n].astype(np.int32) * (1 << Q_FRAC))).clip(min=0)
    # Actually the cleaner comparison is computing pre-comp directly:
    # re-run extract with bfp_compensate=False.
    ext_pre = GoldenLegacyExtractor(bfp_compensate=False)
    pre_feats = ext_pre.extract(samples).astype(np.float32) / (1 << Q_FRAC)
    d_pre = np.abs(pre_feats[:, :n] - rtl_pre_cmp)
    print(f"\n[PRE-comp log features]   max|Δ|={d_pre.max():.4f}  "
          f"mean|Δ|={d_pre.mean():.4f}  median|Δ|={np.median(d_pre):.4f}")

    # Per-frame peak bin for the first few frames
    print("\n[Per-frame peak bins]")
    print(f"{'k':>3} {'rtl_pk':>7} {'gld_pk':>7} {'rtl_bfp':>8} {'gld_bfp':>8}")
    for k in range(min(15, n)):
        print(f"{k:3d} {int(np.argmax(rtl_mag[k])):7d} "
              f"{int(np.argmax(gld_mag[k])):7d} "
              f"{int(rtl_bfp[k]):8d} {int(gld_bfp[k]):8d}")


if __name__ == "__main__":
    main()
