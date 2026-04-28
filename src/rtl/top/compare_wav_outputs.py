"""
compare_wav_outputs.py
======================
Compare RTL speech features against the full-pipeline golden model
for real .wav files saved by test_full_pipeline_speech.

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

_TOP_DIR  = os.path.dirname(os.path.abspath(__file__))
_ML_DIR   = os.path.normpath(os.path.join(_TOP_DIR, "..", "..", "ml"))
_PIPE_DIR = os.path.join(_ML_DIR, "Pipeline")
for p in (_ML_DIR, _PIPE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from golden_model import (
    FullPipelineGoldenExtractor,
    SAMPLE_RATE, SAMPLE_W, N_MELS, N_FFT, Q_FRAC
)

SAMPLE_MAX    = (1 << (SAMPLE_W - 1)) - 1
STARTUP_LOSS  = 0
N_SAMPLES     = 7500    # must match test_full_pipeline_top.py N_PCM_SAMPLES
WAV_DURATION_S = N_SAMPLES / SAMPLE_RATE   # 0.46875 s



def load_wav_as_pcm(path: str) -> np.ndarray:
    """
    Load a .wav file and return int32 PCM at SAMPLE_RATE=16000 Hz,
    trimmed or zero-padded to exactly N_SAMPLES samples.

    Handles: mono/stereo, any sample rate (resampled), int16/int32/float encodings.
    Requires scipy or soundfile:
        pip install scipy --break-system-packages
    """
    data = rate = None
    try:
        from scipy.io import wavfile as _wf
        rate, data = _wf.read(path)
    except Exception:
        pass

    if data is None:
        try:
            import soundfile as _sf
            data, rate = _sf.read(path)
        except Exception as e:
            raise RuntimeError(
                "Cannot load %s.\n"
                "Install scipy:   pip install scipy --break-system-packages\n"
                "or soundfile:    pip install soundfile --break-system-packages\n"
                "Error: %s" % (path, e)
            )

    # Normalise to float64 [-1, 1]
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

    # Resample to SAMPLE_RATE if needed
    if rate != SAMPLE_RATE:
        try:
            from scipy.signal import resample_poly
            from math import gcd
            g     = gcd(int(SAMPLE_RATE), int(rate))
            pcm_f = resample_poly(pcm_f, SAMPLE_RATE // g, rate // g)
        except ImportError:
            n_out = int(round(len(pcm_f) * SAMPLE_RATE / rate))
            pcm_f = np.interp(
                np.linspace(0, 1, n_out),
                np.linspace(0, 1, len(pcm_f)),
                pcm_f,
            )

    # Trim or zero-pad to N_SAMPLES
    if len(pcm_f) >= N_SAMPLES:
        pcm_f = pcm_f[:N_SAMPLES]
    else:
        pcm_f = np.concatenate([pcm_f, np.zeros(N_SAMPLES - len(pcm_f))])

    return np.clip(np.round(pcm_f * SAMPLE_MAX), -SAMPLE_MAX - 1, SAMPLE_MAX
                  ).astype(np.int32)


def find_rtl_npy_files(rtl_dir: str) -> list:
    """Find rtl_features_<stem>.npy files, skipping the baseline test files."""
    skip = {
        "rtl_features.npy",
        "rtl_features_silence.npy",
        "rtl_features_tone.npy",
        "rtl_features_yes.npy",
    }
    matches = []
    for fname in sorted(os.listdir(rtl_dir)):
        if fname.startswith("rtl_features_") and fname.endswith(".npy") \
                and fname not in skip:
            matches.append(os.path.join(rtl_dir, fname))
    return matches


def stem_from_npy(npy_path: str) -> str:
    """Extract the stem from rtl_features_<stem>.npy."""
    return os.path.basename(npy_path)[len("rtl_features_"):-len(".npy")]


def find_wav_for_stem(stem: str, wav_dir: str) -> str | None:
    """
    Find a .wav whose sanitised stem (spaces->_, first 40 chars) matches.
    The testbench uses:  stem = wav_path.stem.replace(' ', '_')[:40]
    """
    for root, _, files in os.walk(wav_dir):
        for fname in files:
            if not fname.lower().endswith(".wav"):
                continue
            wav_stem = os.path.splitext(fname)[0].replace(" ", "_")[:40]
            if wav_stem == stem:
                return os.path.join(root, fname)
    return None


def load_rtl_mat(npy_path: str) -> np.ndarray:
    """Load RTL .npy, normalise to float32 log2, ensure (N_MELS, n_frames)."""
    mat = np.load(npy_path)
    if mat.dtype in (np.uint16, np.int16, np.uint32, np.int32):
        mat = mat.astype(np.float32) / (1 << Q_FRAC)
    elif mat.dtype == np.float64:
        mat = mat.astype(np.float32)
    # Fix orientation if saved transposed
    if mat.ndim == 2 and mat.shape[0] != N_MELS and mat.shape[1] == N_MELS:
        mat = mat.T
    return mat


def plot_comparison(rtl_mat: np.ndarray,
                    golden_mat: np.ndarray,
                    title: str,
                    out_path: str) -> dict:
    """
    Three-panel plot: RTL | Golden | |RTL - Golden|.
    Returns accuracy metrics dict.
    """
    n_frames = min(rtl_mat.shape[1], golden_mat.shape[1])
    rtl   = rtl_mat[:,    STARTUP_LOSS : STARTUP_LOSS + n_frames]
    gold  = golden_mat[:, STARTUP_LOSS : STARTUP_LOSS + n_frames]
    delta = np.abs(rtl - gold)

    vmin = min(rtl.min(), gold.min())
    vmax = max(rtl.max(), gold.max())

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
    kw = dict(aspect="auto", origin="lower", interpolation="nearest", cmap="magma")

    im = axes[0].imshow(rtl, **kw, vmin=vmin, vmax=vmax)
    axes[0].set_title("RTL pipeline  (%d frames)  range=[%.1f, %.1f] log2"
                       % (n_frames, rtl.min(), rtl.max()))
    axes[0].set_ylabel("Mel bin")
    fig.colorbar(im, ax=axes[0], label="log2 energy")

    im = axes[1].imshow(gold, **kw, vmin=vmin, vmax=vmax)
    axes[1].set_title("Full-pipeline golden  (CIC + compFIR + STFFT)")
    axes[1].set_ylabel("Mel bin")
    fig.colorbar(im, ax=axes[1], label="log2 energy")

    im = axes[2].imshow(delta, aspect="auto", origin="lower",
                        interpolation="nearest", cmap="hot",
                        vmin=0, vmax=max(float(delta.max()), 0.01))
    axes[2].set_title("|RTL - Golden|  max=%.4f log2  mean=%.4f log2"
                      % (float(delta.max()), float(delta.mean())))
    axes[2].set_xlabel("Frame index")
    axes[2].set_ylabel("Mel bin")
    fig.colorbar(im, ax=axes[2], label="|delta| log2")

    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    return dict(
        n_frames    = n_frames,
        max_delta   = float(delta.max()),
        mean_delta  = float(delta.mean()),
        rtl_mean    = float(rtl.mean()),
        golden_mean = float(gold.mean()),
        frames_diff = int((delta.max(axis=0) > 0.5).sum()),
    )


def plot_summary(results: list, out_path: str):
    """Bar chart of mean delta per file."""
    if not results:
        return
    names  = [r["stem"] for r in results]
    deltas = [r["mean_delta"] for r in results]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.4 + 2), 4),
                           constrained_layout=True)
    bars = ax.bar(names, deltas, color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("File")
    ax.set_ylabel("Mean |RTL - Golden| (log2)")
    ax.set_title("RTL vs Full-pipeline golden -- per-file accuracy")
    ax.tick_params(axis="x", rotation=35)
    for bar, val in zip(bars, deltas):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                "%.3f" % val, ha="center", va="bottom", fontsize=8)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print("Summary plot -> %s" % out_path)



def main():
    parser = argparse.ArgumentParser(
        description="Compare RTL speech features against full-pipeline golden model"
    )
    parser.add_argument(
        "--rtl_dir", default=_TOP_DIR,
        help="Directory containing rtl_features_<stem>.npy  (default: script dir)"
    )
    parser.add_argument(
        "--wav_dir", default=os.path.join(_TOP_DIR, "speech_data"),
        help="Directory containing the original .wav training files"
    )
    parser.add_argument(
        "--out_dir", default=None,
        help="Where to save plots  (default: same as --rtl_dir)"
    )
    args = parser.parse_args()

    out_dir = args.out_dir or args.rtl_dir
    os.makedirs(out_dir, exist_ok=True)

    # Discover RTL feature files
    npy_files = find_rtl_npy_files(args.rtl_dir)
    if not npy_files:
        print("No rtl_features_*.npy files found in %s" % args.rtl_dir)
        print("Run the cocotb speech test first (test_full_pipeline_speech).")
        sys.exit(0)

    print("Found %d RTL feature file(s)" % len(npy_files))
    print("WAV directory : %s" % args.wav_dir)
    print("Output        : %s" % out_dir)
    print()

    # Instantiate golden model once  (loads SRAM hex files)
    print("Loading FullPipelineGoldenExtractor...")
    golden_model = FullPipelineGoldenExtractor(bfp_compensate=True)
    print("Ready.")
    print()

    all_results = []

    for npy_path in npy_files:
        stem = stem_from_npy(npy_path)
        print("=" * 60)
        print("File : %s" % os.path.basename(npy_path))

        # Load RTL features
        try:
            rtl_mat = load_rtl_mat(npy_path)
        except Exception as e:
            print("  SKIP -- cannot load RTL npy: %s" % e)
            continue
        print("  RTL    : shape=%s  range=[%.2f, %.2f] log2"
              % (rtl_mat.shape, rtl_mat.min(), rtl_mat.max()))

        # Find matching .wav
        wav_path = find_wav_for_stem(stem, args.wav_dir)
        if wav_path is None:
            print("  SKIP -- no matching .wav for stem '%s' in %s"
                  % (stem, args.wav_dir))
            print("  Tip: wav_stem = wav.stem.replace(' ','_')[:40]")
            continue
        print("  WAV    : %s" % os.path.basename(wav_path))

        # Load wav and run through golden model
        try:
            pcm = load_wav_as_pcm(wav_path)
        except RuntimeError as e:
            print("  SKIP -- %s" % e)
            continue

        rms = float(np.sqrt(np.mean(pcm.astype(np.float64)**2)))
        print("  PCM    : %d samples  RMS=%.0f  peak=%d"
              % (len(pcm), rms, int(np.abs(pcm).max())))

        golden_q = golden_model.extract(pcm)
        golden_f = golden_q.astype(np.float32) / (1 << Q_FRAC)
        print("  Golden : shape=%s  range=[%.2f, %.2f] log2"
              % (golden_f.shape, golden_f.min(), golden_f.max()))

        # Plot and compute metrics
        plot_path = os.path.join(out_dir, "comparison_%s.png" % stem)
        metrics   = plot_comparison(
            rtl_mat, golden_f,
            title    = "Speech comparison: %s" % os.path.basename(wav_path),
            out_path = plot_path,
        )
        metrics["stem"] = stem
        all_results.append(metrics)

        print("  Plot   : %s" % plot_path)
        print("  Delta  : max=%.4f log2  mean=%.4f log2  frames_off=%d"
              % (metrics["max_delta"], metrics["mean_delta"], metrics["frames_diff"]))

    # Summary table
    print()
    print("*" * 68)
    print("SUMMARY")
    print("*" * 68)

    if not all_results:
        print("No files compared -- check RTL .npy files and WAV directory.")
        return

    print("%-40s  %9s  %9s  %10s"
          % ("File", "Max delta", "Mean delta", "Frames off"))
    print("-" * 68)
    for r in all_results:
        print("%-40s  %9.4f  %9.4f  %10d"
              % (r["stem"][:40], r["max_delta"], r["mean_delta"], r["frames_diff"]))

    avg_max  = float(np.mean([r["max_delta"]  for r in all_results]))
    avg_mean = float(np.mean([r["mean_delta"] for r in all_results]))
    print("-" * 68)
    print("%-40s  %9.4f  %9.4f" % ("AVERAGE", avg_max, avg_mean))
    print()

    plot_summary(all_results, os.path.join(out_dir, "comparison_summary.png"))

    # Accuracy verdict
    print("Accuracy assessment (mean delta across all files):")
    if avg_mean < 0.5:
        print("  EXCELLENT  -- mean delta < 0.5 log2")
    elif avg_mean < 2.0:
        print("  GOOD       -- mean delta < 2.0 log2  (minor rounding differences)")
    elif avg_mean < 6.0:
        print("  ACCEPTABLE -- mean delta < 6.0 log2")
        print("  Check: fir_trunc [30:15] and CIC truncation [33:18] in full_pipeline_top.sv")
    else:
        print("  POOR       -- mean delta > 6.0 log2")
        print("  Check: fir_trunc bit range, bfpexp compensation, CIC REG_WIDTH")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
