#Compare RTL pipeline output against golden model and torchaudio (features.py).

import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_TOP_DIR  = os.path.dirname(os.path.abspath(__file__))
_ML_DIR   = os.path.normpath(os.path.join(_TOP_DIR, "..", "..", "ml"))
_PIPE_DIR = os.path.join(_ML_DIR, "Pipeline")
for p in (_ML_DIR, _PIPE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from golden_model import GoldenExtractor, SAMPLE_RATE, SAMPLE_W, N_MELS, N_FFT, Q_FRAC
from features import LogMelExtractor

SAMPLE_MAX   = (1 << (SAMPLE_W - 1)) - 1
STARTUP_LOSS = 3      # RTL drops first 3 frames (FFT pipeline fill)
N_SAMPLES    = 7500   # must match test_pipeline_top.py


def make_chirp(n: int) -> np.ndarray:
    #200-7000 Hz chirp, same as test_pipeline_top.py.
    dur = n / SAMPLE_RATE
    t = np.arange(n) / SAMPLE_RATE
    phase = 2 * np.pi * (200 * t + (7000 - 200) / (2 * dur) * t**2)
    return (np.sin(phase) * SAMPLE_MAX).astype(np.int32)


def main():
    parser = argparse.ArgumentParser(description="Compare RTL vs golden vs torchaudio")
    parser.add_argument("--rtl", type=str, default=os.path.join(_TOP_DIR, "rtl_features.npy"),
                        help="RTL features .npy from test_pipeline_top.py")
    args = parser.parse_args()

    samples = make_chirp(N_SAMPLES)
    print(f"Generated chirp: {len(samples)} samples, {len(samples)/SAMPLE_RATE:.2f}s")

    # Load RTL output
    if os.path.exists(args.rtl):
        rtl_mat = np.load(args.rtl)
        print(f"RTL features loaded: {args.rtl}  shape={rtl_mat.shape}")
    else:
        print(f"WARNING: {args.rtl} not found. Run 'make test-cocotb' first.")
        rtl_mat = None

    # Golden model — bit-accurate fixed-point replica
    golden = GoldenExtractor()
    golden_q12 = golden.extract(samples)                        # (N_MELS, n_frames) uint16
    golden_float = golden_q12.astype(np.float32) / (1 << Q_FRAC)
    print(f"Golden model: {golden_float.shape}  ({golden_float.shape[1]} frames)")

    # Torchaudio floating-point reference
    extractor = LogMelExtractor(
        sample_rate=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS,
        hop_length=N_FFT // 2, window_length=N_FFT,
    )
    feat_spec, _ = extractor.extract(samples)
    print(f"features.py:  {feat_spec.shape}  ({feat_spec.shape[1]} frames)")

    # Align frames: RTL drops STARTUP_LOSS frames at the start
    print("\n" + "=" * 60)

    if rtl_mat is not None:
        n_rtl  = rtl_mat.shape[1]
        n_cmp  = min(n_rtl, golden_float.shape[1] - STARTUP_LOSS)
        rtl_cmp    = rtl_mat[:, :n_cmp]
        golden_cmp = golden_float[:, STARTUP_LOSS:STARTUP_LOSS + n_cmp]
        diff_rg = np.abs(rtl_cmp - golden_cmp)

        print(f"\nRTL vs Golden ({n_cmp} frames, golden offset by {STARTUP_LOSS})")
        print(f"  Max delta  : {diff_rg.max():.6f} log2")
        print(f"  Mean delta : {diff_rg.mean():.6f} log2")
        print(f"  Frames with delta > 0: {(diff_rg.max(axis=0) > 0).sum()} / {n_cmp}")

        n_cmp2 = min(n_rtl, feat_spec.shape[1] - STARTUP_LOSS)
        rtl_cmp2  = rtl_mat[:, :n_cmp2]
        feat_cmp2 = feat_spec[:, STARTUP_LOSS:STARTUP_LOSS + n_cmp2]
        diff_rf = np.abs(rtl_cmp2 - feat_cmp2)

        print(f"\nRTL vs torchaudio ({n_cmp2} frames, offset by {STARTUP_LOSS})")
        print(f"  Max delta  : {diff_rf.max():.6f} log2")
        print(f"  Mean delta : {diff_rf.mean():.6f} log2")

    n_cmp3 = min(golden_float.shape[1], feat_spec.shape[1])
    diff_gf = np.abs(golden_float[:, :n_cmp3] - feat_spec[:, :n_cmp3])

    print(f"\nGolden vs torchaudio ({n_cmp3} frames)")
    print(f"  Max delta  : {diff_gf.max():.6f} log2")
    print(f"  Mean delta : {diff_gf.mean():.6f} log2")

    # Spectrogram comparison plot (aligned to RTL frame range)
    has_rtl = rtl_mat is not None
    if has_rtl:
        golden_plot = golden_float[:, STARTUP_LOSS:STARTUP_LOSS + rtl_mat.shape[1]]
        feat_plot   = feat_spec[:, STARTUP_LOSS:STARTUP_LOSS + rtl_mat.shape[1]]
    else:
        golden_plot = golden_float
        feat_plot   = feat_spec

    n_panels = 4 if has_rtl else 3
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3.5 * n_panels), constrained_layout=True)
    kw = dict(aspect="auto", origin="lower", interpolation="nearest", cmap="magma")

    all_data = [golden_plot, feat_plot] + ([rtl_mat] if has_rtl else [])
    vmin = min(d.min() for d in all_data)
    vmax = max(d.max() for d in all_data)

    idx = 0
    if has_rtl:
        im = axes[idx].imshow(rtl_mat, **kw, vmin=vmin, vmax=vmax)
        axes[idx].set_title(f"RTL pipeline_top ({rtl_mat.shape[1]} frames)")
        axes[idx].set_ylabel("Mel bin")
        fig.colorbar(im, ax=axes[idx], label="log2 energy")
        idx += 1

    im = axes[idx].imshow(golden_plot, **kw, vmin=vmin, vmax=vmax)
    axes[idx].set_title(f"Golden model ({golden_plot.shape[1]} frames, aligned)")
    axes[idx].set_ylabel("Mel bin")
    fig.colorbar(im, ax=axes[idx], label="log2 energy")
    idx += 1

    im = axes[idx].imshow(feat_plot, **kw, vmin=vmin, vmax=vmax)
    axes[idx].set_title(f"torchaudio ({feat_plot.shape[1]} frames, aligned)")
    axes[idx].set_ylabel("Mel bin")
    fig.colorbar(im, ax=axes[idx], label="log2 energy")
    idx += 1

    # Delta heatmap
    if has_rtl:
        im = axes[idx].imshow(diff_rg, aspect="auto", origin="lower",
                              interpolation="nearest", cmap="hot",
                              vmin=0, vmax=max(diff_rg.max(), 0.01))
        axes[idx].set_title(f"|RTL - Golden|  (max={diff_rg.max():.4f} log2)")
    else:
        im = axes[idx].imshow(diff_gf, aspect="auto", origin="lower",
                              interpolation="nearest", cmap="hot",
                              vmin=0, vmax=max(diff_gf.max(), 0.01))
        axes[idx].set_title(f"|Golden - torchaudio|  (max={diff_gf.max():.4f} log2)")
    axes[idx].set_xlabel("Frame index")
    axes[idx].set_ylabel("Mel bin")
    fig.colorbar(im, ax=axes[idx], label="|delta| log2")

    out_path = os.path.join(_TOP_DIR, "comparison.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\nPlot saved -> {out_path}")


if __name__ == "__main__":
    main()
