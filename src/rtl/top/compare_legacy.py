"""Generate comparison_legacy.png — Legacy RTL vs GoldenLegacyExtractor.

Visualises how closely the legacy golden model replicates the pre-FIFO RTL
pipeline.  Run after the legacy cocotb snapshot (rtl_*_legacy.npy) has been
saved; uses src/ml/golden_model_legacy.py as the reference.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    rtl = np.load(os.path.join(_TOP, "rtl_features_legacy.npy"))  # (40, 55) float32 log2
    print(f"Legacy RTL features: {rtl.shape}")

    samples = make_chirp(N_SAMPLES)
    ext = GoldenLegacyExtractor()
    gld_q = ext.extract(samples)                              # (40, 55) uint16 Q_FRAC
    gld = gld_q.astype(np.float32) / (1 << Q_FRAC)           # float32 log2
    print(f"Legacy golden features: {gld.shape}")

    n = min(rtl.shape[1], gld.shape[1])
    rtl_cmp, gld_cmp = rtl[:, :n], gld[:, :n]
    diff = np.abs(rtl_cmp - gld_cmp)

    print(f"\nComparison stats ({n} frames):")
    print(f"  max |Δ|    : {diff.max():.4f} log2")
    print(f"  mean |Δ|   : {diff.mean():.4f} log2")
    print(f"  median |Δ| : {np.median(diff):.4f} log2")

    # Also compute stats excluding the known-anomalous startup frame 0
    diff_noF0 = diff[:, 1:]
    print(f"\nExcluding Bug-C startup frame 0:")
    print(f"  max |Δ|    : {diff_noF0.max():.4f} log2")
    print(f"  mean |Δ|   : {diff_noF0.mean():.4f} log2")
    print(f"  median |Δ| : {np.median(diff_noF0):.4f} log2")

    # 4-panel plot: RTL, Golden, |Δ| with full range, |Δ| zoomed to the
    # typical (non-f0) residual.
    fig, axes = plt.subplots(4, 1, figsize=(13, 13), constrained_layout=True)
    vmin = min(rtl_cmp.min(), gld_cmp.min())
    vmax = max(rtl_cmp.max(), gld_cmp.max())
    kw = dict(aspect="auto", origin="lower", interpolation="nearest", cmap="magma")

    im0 = axes[0].imshow(rtl_cmp, **kw, vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Legacy RTL pipeline_top ({n} frames) — "
                      f"hop drifts 129/148 (mean 138.5), not 128")
    axes[0].set_ylabel("Mel bin")
    fig.colorbar(im0, ax=axes[0], label="log2 energy")

    im1 = axes[1].imshow(gld_cmp, **kw, vmin=vmin, vmax=vmax)
    axes[1].set_title("GoldenLegacyExtractor — models Bug A (sample drop), "
                      "Bug B (Hann rotation), Bug D (3-bin cyclic roll)")
    axes[1].set_ylabel("Mel bin")
    fig.colorbar(im1, ax=axes[1], label="log2 energy")

    im2 = axes[2].imshow(diff, aspect="auto", origin="lower",
                          interpolation="nearest", cmap="hot",
                          vmin=0, vmax=max(diff.max(), 0.01))
    axes[2].set_title(f"|RTL − Golden|   max={diff.max():.2f} log2, "
                      f"mean={diff.mean():.3f}, median={np.median(diff):.3f}")
    axes[2].set_ylabel("Mel bin")
    fig.colorbar(im2, ax=axes[2], label="|Δ| log2")

    # Same diff, saturated at 2 log2 to visualise the 99th-percentile residual.
    im3 = axes[3].imshow(diff, aspect="auto", origin="lower",
                          interpolation="nearest", cmap="hot",
                          vmin=0, vmax=2.0)
    axes[3].set_title("|RTL − Golden| (clipped at 2.0 log2 to show the bulk)"
                      " — dark = bit-exact")
    axes[3].set_xlabel("Frame index")
    axes[3].set_ylabel("Mel bin")
    fig.colorbar(im3, ax=axes[3], label="|Δ| log2  (clipped)")

    out = os.path.join(_TOP, "comparison_legacy.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
