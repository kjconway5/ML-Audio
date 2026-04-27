"""
compare_wav_outputs.py
======================
Compare RTL features against the full-pipeline golden model for real .wav files.

The test_full_pipeline_speech cocotb test saves:
    rtl_features_<stem>.npy   (N_MELS, n_frames)  float32 log2

This script:
  1. Discovers those .npy files in the test directory
  2. Finds the matching .wav file in SPEECH_WAV_DIR
  3. Runs the wav through FullPipelineGoldenExtractor (CIC + FIR + STFFT)
  4. Plots: RTL | Golden | |RTL - Golden|
  5. Prints a per-file and summary accuracy table

Usage:
    python3 compare_wav_outputs.py
    python3 compare_wav_outputs.py --wav_dir /path/to/speech_data
    python3 compare_wav_outputs.py --rtl_dir /path/to/rtl/top --wav_dir /data/yes
"""

import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths -- same layout as the testbench
# ---------------------------------------------------------------------------
_THIS_DIR  = Path(__file__).resolve().parent
_ML_DIR    = (_THIS_DIR / ".." / ".." / "ml").resolve()
_PIPE_DIR  = _ML_DIR / "Pipeline"

for _p in [str(_ML_DIR), str(_PIPE_DIR), str(_THIS_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Constants (must match test + RTL parameters)
# ---------------------------------------------------------------------------
PCM_RATE      = 16_000
N_FFT         = 256
N_MELS        = 40
Q_FRAC        = 10
WAV_DURATION_S = 0.47     # seconds -- same as testbench WAV_DURATION_S
DECIM          = 63

# ---------------------------------------------------------------------------
# Lazy imports to avoid circular dependency
# ---------------------------------------------------------------------------

def _import_golden():
    """Import FullPipelineGoldenExtractor lazily after path is set up."""
    try:
        from full_pipeline_golden import FullPipelineGoldenExtractor
        return FullPipelineGoldenExtractor
    except ImportError as e:
        print("ERROR: Cannot import FullPipelineGoldenExtractor.")
        print("  Make sure full_pipeline_golden.py is in the same directory.")
        print("  Original error:", e)
        sys.exit(1)


# ---------------------------------------------------------------------------
# WAV loading (same as testbench _load_wav)
# ---------------------------------------------------------------------------

def load_wav_as_pcm(path: Path) -> np.ndarray:
    """Load .wav and return int32 PCM at PCM_RATE, length WAV_DURATION_S * PCM_RATE."""
    data = rate = None
    try:
        from scipy.io import wavfile as _wf
        rate, data = _wf.read(str(path))
    except Exception:
        pass

    if data is None:
        try:
            import soundfile as _sf
            data, rate = _sf.read(str(path))
        except Exception as e:
            raise RuntimeError("Cannot load %s: %s" % (path, e))

    # Normalise to float64 in [-1, 1]
    dt = np.asarray(data).dtype
    if dt == np.int16:
        pcm_f = data.astype(np.float64) / 32768.0
    elif dt == np.int32:
        pcm_f = data.astype(np.float64) / 2147483648.0
    elif dt == np.uint8:
        pcm_f = (data.astype(np.float64) - 128.0) / 128.0
    else:
        pcm_f = np.asarray(data, dtype=np.float64)

    # Stereo -> mono
    if pcm_f.ndim == 2:
        pcm_f = pcm_f.mean(axis=1)

    # Resample to PCM_RATE
    if rate != PCM_RATE:
        try:
            from scipy.signal import resample_poly
            from math import gcd
            g     = gcd(int(PCM_RATE), int(rate))
            pcm_f = resample_poly(pcm_f, PCM_RATE // g, rate // g)
        except ImportError:
            n_out = int(round(len(pcm_f) * PCM_RATE / rate))
            pcm_f = np.interp(
                np.linspace(0, 1, n_out),
                np.linspace(0, 1, len(pcm_f)),
                pcm_f,
            )

    # Trim / zero-pad
    target = int(round(WAV_DURATION_S * PCM_RATE))
    if len(pcm_f) >= target:
        pcm_f = pcm_f[:target]
    else:
        pcm_f = np.concatenate([pcm_f, np.zeros(target - len(pcm_f))])

    return np.clip(np.round(pcm_f * 32767), -32768, 32767).astype(np.int32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_rtl_npy_files(rtl_dir: Path) -> list:
    """Return sorted list of rtl_features_*.npy paths (excluding chirp/silence/tone)."""
    skip = {"rtl_features.npy", "rtl_features_silence.npy",
            "rtl_features_tone.npy", "rtl_features_yes.npy"}
    return sorted(
        p for p in rtl_dir.glob("rtl_features_*.npy")
        if p.name not in skip
    )


def stem_to_wav(stem: str, wav_dir: Path) -> Path | None:
    """Find a .wav in wav_dir whose stem (first 40 chars, spaces->_) matches stem."""
    for wav in wav_dir.rglob("*.wav"):
        candidate = wav.stem.replace(" ", "_")[:40]
        if candidate == stem:
            return wav
    return None


def load_rtl_mat(npy_path: Path) -> np.ndarray:
    """Load RTL .npy, ensure shape (N_MELS, n_frames), normalise to float32 log2."""
    mat = np.load(str(npy_path))
    if mat.dtype in (np.uint16, np.int16, np.uint32, np.int32):
        mat = mat.astype(np.float32) / (1 << Q_FRAC)
    elif mat.dtype == np.float64:
        mat = mat.astype(np.float32)
    # Fix orientation if saved transposed
    if mat.ndim == 2 and mat.shape[0] != N_MELS and mat.shape[1] == N_MELS:
        mat = mat.T
    return mat


# ---------------------------------------------------------------------------
# Per-file comparison and plot
# ---------------------------------------------------------------------------

def compare_one(rtl_mat: np.ndarray,
                golden_mat: np.ndarray,
                title: str,
                out_path: Path) -> dict:
    """
    Plot RTL | Golden | |RTL - Golden| for one file.
    Returns dict of accuracy metrics.
    """
    n_frames = min(rtl_mat.shape[1], golden_mat.shape[1])
    rtl   = rtl_mat[:, :n_frames]
    gold  = golden_mat[:, :n_frames]
    delta = np.abs(rtl - gold)

    # Shared colour scale for RTL and Golden panels
    vmin = min(rtl.min(), gold.min())
    vmax = max(rtl.max(), gold.max())

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
    kw = dict(aspect="auto", origin="lower", interpolation="nearest", cmap="magma")

    im0 = axes[0].imshow(rtl, **kw, vmin=vmin, vmax=vmax)
    axes[0].set_title("RTL pipeline  (%d frames)  range=[%.1f, %.1f] log2"
                      % (n_frames, rtl.min(), rtl.max()))
    axes[0].set_ylabel("Mel bin")
    fig.colorbar(im0, ax=axes[0], label="log2 energy")

    im1 = axes[1].imshow(gold, **kw, vmin=vmin, vmax=vmax)
    axes[1].set_title("Full-pipeline golden  (CIC + compFIR + STFFT)")
    axes[1].set_ylabel("Mel bin")
    fig.colorbar(im1, ax=axes[1], label="log2 energy")

    im2 = axes[2].imshow(delta, aspect="auto", origin="lower",
                         interpolation="nearest", cmap="hot",
                         vmin=0, vmax=max(delta.max(), 0.01))
    axes[2].set_title("|RTL - Golden|  max=%.4f log2  mean=%.4f log2"
                      % (delta.max(), delta.mean()))
    axes[2].set_xlabel("Frame index")
    axes[2].set_ylabel("Mel bin")
    fig.colorbar(im2, ax=axes[2], label="|delta| log2")

    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)

    return {
        "n_frames":    n_frames,
        "max_delta":   float(delta.max()),
        "mean_delta":  float(delta.mean()),
        "rtl_mean":    float(rtl.mean()),
        "golden_mean": float(gold.mean()),
        "frames_diff": int((delta.max(axis=0) > 0.5).sum()),
    }


# ---------------------------------------------------------------------------
# Summary plot: one column per file
# ---------------------------------------------------------------------------

def plot_summary(results: list, out_path: Path):
    """Bar chart of mean delta per file."""
    if not results:
        return
    names  = [r["stem"] for r in results]
    deltas = [r["mean_delta"] for r in results]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2 + 2), 4),
                           constrained_layout=True)
    bars = ax.bar(names, deltas, color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("File")
    ax.set_ylabel("Mean |RTL - Golden| (log2)")
    ax.set_title("RTL vs Full-pipeline golden -- per-file mean delta")
    ax.tick_params(axis="x", rotation=35)

    for bar, val in zip(bars, deltas):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                "%.3f" % val, ha="center", va="bottom", fontsize=8)

    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)
    print("Summary plot -> %s" % out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare RTL speech features against full-pipeline golden model"
    )
    parser.add_argument(
        "--rtl_dir", default=str(_THIS_DIR),
        help="Directory containing rtl_features_<stem>.npy files (default: script dir)"
    )
    parser.add_argument(
        "--wav_dir",
        default=str(_THIS_DIR / "speech_data"),
        help="Directory containing the original .wav files"
    )
    parser.add_argument(
        "--out_dir", default=None,
        help="Where to save plots (default: same as --rtl_dir)"
    )
    args = parser.parse_args()

    rtl_dir = Path(args.rtl_dir)
    wav_dir = Path(args.wav_dir)
    out_dir = Path(args.out_dir) if args.out_dir else rtl_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Import golden model (lazy to avoid circular imports)
    GoldenCls = _import_golden()

    # Find RTL feature files
    npy_files = find_rtl_npy_files(rtl_dir)
    if not npy_files:
        print("No rtl_features_*.npy files found in %s" % rtl_dir)
        print("Run the cocotb test first:")
        print("  make test-cocotb  (from the top/ directory)")
        sys.exit(0)

    print("Found %d RTL feature file(s) in %s" % (len(npy_files), rtl_dir))
    print("WAV directory: %s" % wav_dir)
    print("Output directory: %s" % out_dir)
    print()

    # Instantiate golden model once (loads SRAMs / hex files)
    print("Loading full-pipeline golden model...")
    golden_model = GoldenCls(bfp_compensate=True)
    print("Golden model ready.")
    print()

    all_results = []

    for npy_path in npy_files:
        # Stem = everything after "rtl_features_"
        stem = npy_path.stem[len("rtl_features_"):]
        print("=" * 60)
        print("File: %s" % npy_path.name)

        # Load RTL features
        try:
            rtl_mat = load_rtl_mat(npy_path)
        except Exception as e:
            print("  SKIP -- cannot load RTL npy: %s" % e)
            continue
        print("  RTL   : shape=%s  range=[%.2f, %.2f] log2"
              % (rtl_mat.shape, rtl_mat.min(), rtl_mat.max()))

        # Find matching wav file
        wav_path = stem_to_wav(stem, wav_dir)
        if wav_path is None:
            print("  SKIP -- no matching .wav found in %s for stem '%s'"
                  % (wav_dir, stem))
            print("  Tip: wav filename stem (spaces->_, first 40 chars) must match.")
            continue
        print("  WAV   : %s" % wav_path.name)

        # Load wav and run through golden model
        try:
            pcm = load_wav_as_pcm(wav_path)
        except RuntimeError as e:
            print("  SKIP -- %s" % e)
            continue

        rms = float(np.sqrt(np.mean(pcm.astype(np.float64)**2)))
        print("  PCM   : %d samples  RMS=%.0f  peak=%d"
              % (len(pcm), rms, int(np.abs(pcm).max())))

        golden_q10 = golden_model.extract(pcm)
        golden_mat = golden_q10.astype(np.float32) / (1 << Q_FRAC)
        print("  Golden: shape=%s  range=[%.2f, %.2f] log2"
              % (golden_mat.shape, golden_mat.min(), golden_mat.max()))

        # Compare and plot
        plot_path = out_dir / ("comparison_%s.png" % stem)
        title     = "Speech comparison: %s" % wav_path.name
        metrics   = compare_one(rtl_mat, golden_mat, title, plot_path)
        metrics["stem"] = stem

        all_results.append(metrics)

        print("  Plot  : %s" % plot_path)
        print("  Delta : max=%.4f log2  mean=%.4f log2  frames_off=%.0f"
              % (metrics["max_delta"], metrics["mean_delta"], metrics["frames_diff"]))

    # Print summary table
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if not all_results:
        print("No files compared -- check RTL npy files and WAV directory.")
        return

    print("%-42s  %8s  %8s  %10s" % ("File", "Max delt", "Mean delt", "Frames off"))
    print("-" * 75)
    for r in all_results:
        print("%-42s  %8.4f  %8.4f  %10d"
              % (r["stem"][:42], r["max_delta"], r["mean_delta"], r["frames_diff"]))

    max_d  = np.mean([r["max_delta"]  for r in all_results])
    mean_d = np.mean([r["mean_delta"] for r in all_results])
    print("-" * 75)
    print("%-42s  %8.4f  %8.4f" % ("AVERAGE", max_d, mean_d))
    print()

    # Summary bar chart
    plot_summary(all_results, out_dir / "comparison_summary.png")

    # Accuracy assessment
    print("Accuracy assessment:")
    if mean_d < 0.5:
        print("  EXCELLENT -- mean delta < 0.5 log2 (RTL closely matches golden)")
    elif mean_d < 2.0:
        print("  GOOD -- mean delta < 2.0 log2 (minor fixed-point rounding differences)")
    elif mean_d < 6.0:
        print("  ACCEPTABLE -- mean delta < 6 log2 (CIC/FIR gain offset expected)")
        print("  Check: fir_trunc bit range in full_pipeline_top.sv")
    else:
        print("  POOR -- mean delta > 6 log2")
        print("  Check: fir_trunc bit range, CIC truncation [33:18], bfpexp compensation")


if __name__ == "__main__":
    main()