"""
compare_wav_outputs.py
======================
Compare RTL speech features against the full-pipeline golden model,
grouped by keyword.

For each keyword (subdirectory of speech_data/):
  - Computes per-file RTL vs golden delta
  - Plots one averaged spectrogram comparison (RTL | Golden | Delta)
  - Reports accuracy as a percentage:
      accuracy = 100 * (1 - mean_delta / max_possible_log2)

Final summary:
  - Bar chart of accuracy % per keyword
  - Full table of per-file metrics
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

SAMPLE_MAX     = (1 << (SAMPLE_W - 1)) - 1
STARTUP_LOSS   = 0
N_SAMPLES      = 7500
WAV_DURATION_S = N_SAMPLES / SAMPLE_RATE

# Maximum possible log2 value (LOG_OUT_W=16 bits, Q_FRAC=10)
MAX_LOG2       = (1 << (16 - Q_FRAC)) - 1 / (1 << Q_FRAC)   # ~63.999


# ---------------------------------------------------------------------------
# WAV loading  (identical to testbench _load_wav)
# ---------------------------------------------------------------------------

def load_wav_as_pcm(path: str) -> np.ndarray:
    """Load .wav -> int32 PCM at SAMPLE_RATE, trimmed/padded to N_SAMPLES."""
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
                "Install scipy:  pip install scipy --break-system-packages\n"
                "Error: %s" % (path, e)
            )

    dt = np.asarray(data).dtype
    if dt == np.int16:
        pcm_f = data.astype(np.float64) / 32768.0
    elif dt == np.int32:
        pcm_f = data.astype(np.float64) / 2147483648.0
    elif dt == np.uint8:
        pcm_f = (data.astype(np.float64) - 128.0) / 128.0
    else:
        pcm_f = np.asarray(data, dtype=np.float64)

    if pcm_f.ndim == 2:
        pcm_f = pcm_f.mean(axis=1)

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

    target = int(round(WAV_DURATION_S * SAMPLE_RATE))
    pcm_f  = pcm_f[:target] if len(pcm_f) >= target \
             else np.concatenate([pcm_f, np.zeros(target - len(pcm_f))])

    return np.clip(np.round(pcm_f * SAMPLE_MAX), -SAMPLE_MAX - 1, SAMPLE_MAX
                   ).astype(np.int32)


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def load_rtl_mat(npy_path: str) -> np.ndarray:
    """Load RTL .npy -> float32 log2, (N_MELS, n_frames)."""
    mat = np.load(npy_path)
    if mat.dtype in (np.uint16, np.int16, np.uint32, np.int32):
        mat = mat.astype(np.float32) / (1 << Q_FRAC)
    elif mat.dtype == np.float64:
        mat = mat.astype(np.float32)
    if mat.ndim == 2 and mat.shape[0] != N_MELS and mat.shape[1] == N_MELS:
        mat = mat.T
    return mat


def parse_npy_name(fname: str):
    """
    Parse rtl_features_<keyword>_<stem>.npy -> (keyword, stem).
    Falls back to (None, stem) for non-speech files.
    """
    base = fname[len("rtl_features_"):-len(".npy")]
    # keyword is the first segment before _
    # stem may itself contain underscores, so split on first _ only up to
    # depth 1 -- but keyword dirs are known so we match greedily.
    # Convention: testbench saves as  <keyword>_<stem>  where keyword has no _.
    # We split on the first _ to get keyword.
    parts = base.split("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return None, base


def find_npy_files(rtl_dir: str) -> dict:
    """
    Scan rtl_dir for rtl_features_<keyword>_<stem>.npy files.
    Returns {keyword: [(npy_path, stem), ...]}
    Skips baseline files (chirp, silence, tone).
    """
    skip = {"chirp", "silence", "tone_1kHz", "yes", "no"}   # old flat stems
    grouped = {}

    for fname in sorted(os.listdir(rtl_dir)):
        if not (fname.startswith("rtl_features_") and fname.endswith(".npy")):
            continue
        keyword, stem = parse_npy_name(fname)
        if keyword in (None,) or stem in skip:
            continue
        # Baseline singletons have no keyword prefix -- skip them
        if fname in ("rtl_features.npy", "rtl_features_silence.npy",
                     "rtl_features_tone_1kHz.npy"):
            continue
        grouped.setdefault(keyword, []).append(
            (os.path.join(rtl_dir, fname), stem)
        )

    return grouped


def find_wav(keyword: str, stem: str, wav_dir: str) -> str:
    """
    Find <wav_dir>/<keyword>/<stem>.wav  (stem may have spaces->_ already applied).
    Also tries wav_dir/<keyword>/<stem with _ -> space>.wav.
    """
    kw_dir = os.path.join(wav_dir, keyword)
    if not os.path.isdir(kw_dir):
        kw_dir = wav_dir   # flat layout fallback

    for root, _, files in os.walk(kw_dir):
        for fname in files:
            if not fname.lower().endswith(".wav"):
                continue
            candidate = os.path.splitext(fname)[0].replace(" ", "_")[:32]
            if candidate == stem:
                return os.path.join(root, fname)
    return None


# ---------------------------------------------------------------------------
# Accuracy metric
# ---------------------------------------------------------------------------

def delta_to_accuracy(mean_delta: float) -> float:
    """
    Convert mean |RTL - Golden| (log2) to an accuracy percentage.

    accuracy = max(0, 100 * (1 - mean_delta / MAX_LOG2))

    A perfect match (mean_delta=0) -> 100%.
    A mean delta equal to the full log2 range -> 0%.
    Values in between scale linearly.
    """
    return max(0.0, 100.0 * (1.0 - mean_delta / MAX_LOG2))


# ---------------------------------------------------------------------------
# Per-keyword averaged spectrogram plot
# ---------------------------------------------------------------------------

def plot_keyword_spectrogram(keyword: str,
                             rtl_mats: list,
                             golden_mats: list,
                             out_path: str) -> dict:
    """
    Average all RTL and golden spectrograms for one keyword, then plot
    three panels: Mean RTL | Mean Golden | Mean |RTL - Golden|.

    All matrices are trimmed to the minimum frame count before averaging
    so that shapes are compatible.

    Returns per-keyword aggregate metrics.
    """
    # Trim to minimum frame count across all files
    min_frames = min(min(m.shape[1] for m in rtl_mats),
                     min(m.shape[1] for m in golden_mats))

    rtl_stack    = np.stack([m[:, :min_frames] for m in rtl_mats],    axis=0)
    golden_stack = np.stack([m[:, :min_frames] for m in golden_mats], axis=0)

    mean_rtl    = rtl_stack.mean(axis=0)       # (N_MELS, min_frames)
    mean_golden = golden_stack.mean(axis=0)
    mean_delta  = np.abs(mean_rtl - mean_golden)

    # Per-sample delta across all files (for per-file statistics)
    all_deltas  = np.abs(rtl_stack - golden_stack)   # (n_files, N_MELS, min_frames)

    vmin = min(float(mean_rtl.min()), float(mean_golden.min()))
    vmax = max(float(mean_rtl.max()), float(mean_golden.max()))

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
    kw_plot = dict(aspect="auto", origin="lower", interpolation="nearest", cmap="magma")

    im = axes[0].imshow(mean_rtl, **kw_plot, vmin=vmin, vmax=vmax)
    axes[0].set_title(
        "RTL  (mean of %d files)  range=[%.1f, %.1f] log2"
        % (len(rtl_mats), float(mean_rtl.min()), float(mean_rtl.max()))
    )
    axes[0].set_ylabel("Mel bin")
    fig.colorbar(im, ax=axes[0], label="log2 energy")

    im = axes[1].imshow(mean_golden, **kw_plot, vmin=vmin, vmax=vmax)
    axes[1].set_title("Golden  (mean of %d files,  CIC + compFIR + STFFT)"
                      % len(golden_mats))
    axes[1].set_ylabel("Mel bin")
    fig.colorbar(im, ax=axes[1], label="log2 energy")

    im = axes[2].imshow(
        mean_delta, aspect="auto", origin="lower",
        interpolation="nearest", cmap="hot",
        vmin=0, vmax=max(float(mean_delta.max()), 0.01)
    )
    acc = delta_to_accuracy(float(all_deltas.mean()))
    axes[2].set_title(
        "Mean |RTL - Golden|  max=%.3f  mean=%.3f log2  accuracy=%.1f%%"
        % (float(mean_delta.max()), float(all_deltas.mean()), acc)
    )
    axes[2].set_xlabel("Frame index")
    axes[2].set_ylabel("Mel bin")
    fig.colorbar(im, ax=axes[2], label="|delta| log2")

    fig.suptitle("Keyword: '%s'" % keyword, fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    return dict(
        keyword     = keyword,
        n_files     = len(rtl_mats),
        min_frames  = min_frames,
        max_delta   = float(all_deltas.max()),
        mean_delta  = float(all_deltas.mean()),
        accuracy_pct = acc,
    )


# ---------------------------------------------------------------------------
# Summary accuracy bar chart
# ---------------------------------------------------------------------------

def plot_accuracy_summary(kw_metrics: list, out_path: str):
    """Bar chart: accuracy % per keyword, sorted descending."""
    kw_metrics_sorted = sorted(kw_metrics, key=lambda r: r["accuracy_pct"], reverse=True)
    names    = [r["keyword"] for r in kw_metrics_sorted]
    accs     = [r["accuracy_pct"] for r in kw_metrics_sorted]
    n_files  = [r["n_files"] for r in kw_metrics_sorted]

    colours = ["#2ecc71" if a >= 90 else "#e67e22" if a >= 70 else "#e74c3c"
               for a in accs]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.4 + 2), 5),
                           constrained_layout=True)
    bars = ax.bar(names, accs, color=colours, edgecolor="black", linewidth=0.5)

    ax.axhline(90, color="green",  linestyle="--", linewidth=1, label="90% target")
    ax.axhline(70, color="orange", linestyle="--", linewidth=1, label="70% threshold")

    for bar, val, n in zip(bars, accs, n_files):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                "%.1f%%\n(n=%d)" % (val, n),
                ha="center", va="bottom", fontsize=8)

    ax.set_ylim(0, 110)
    ax.set_xlabel("Keyword")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(
        "RTL vs Full-pipeline golden -- accuracy per keyword\n"
        "accuracy = 100 x (1 - mean_delta / max_log2)"
    )
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=8)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print("Accuracy summary plot -> %s" % out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Per-keyword RTL vs golden spectrogram comparison"
    )
    parser.add_argument(
        "--rtl_dir", default=_TOP_DIR,
        help="Directory with rtl_features_<keyword>_<stem>.npy files"
    )
    parser.add_argument(
        "--wav_dir", default=os.path.join(_TOP_DIR, "speech_data"),
        help="Root of the speech dataset (one subdirectory per keyword)"
    )
    parser.add_argument(
        "--out_dir", default=None,
        help="Where to save plots  (default: same as --rtl_dir)"
    )
    args = parser.parse_args()

    out_dir = args.out_dir or args.rtl_dir
    os.makedirs(out_dir, exist_ok=True)

    # Discover grouped npy files
    grouped = find_npy_files(args.rtl_dir)
    if not grouped:
        print("No keyword-grouped rtl_features_*.npy found in %s" % args.rtl_dir)
        print("Run test_full_pipeline_speech first.")
        sys.exit(0)

    total_files = sum(len(v) for v in grouped.values())
    print("Keywords found: %s" % ", ".join(sorted(grouped)))
    print("Total RTL feature files: %d" % total_files)
    print("WAV directory: %s" % args.wav_dir)
    print("Output:        %s" % out_dir)
    print()

    # Load golden model once
    print("Loading FullPipelineGoldenExtractor ...")
    golden_model = FullPipelineGoldenExtractor(bfp_compensate=True)
    print("Ready.")
    print()

    kw_summary   = []   # one entry per keyword
    all_file_rows = []  # one entry per file (for full table)

    for keyword in sorted(grouped):
        entries = grouped[keyword]   # [(npy_path, stem), ...]
        print("=" * 64)
        print("Keyword: '%s'  (%d files)" % (keyword, len(entries)))
        print("=" * 64)

        kw_rtl_mats    = []
        kw_golden_mats = []

        for npy_path, stem in entries:
            # Load RTL features
            try:
                rtl_mat = load_rtl_mat(npy_path)
            except Exception as e:
                print("  [%s] SKIP RTL load: %s" % (stem, e))
                continue

            # Find and load the matching wav
            wav_path = find_wav(keyword, stem, args.wav_dir)
            if wav_path is None:
                print("  [%s] SKIP -- no matching .wav" % stem)
                continue

            try:
                pcm = load_wav_as_pcm(wav_path)
            except RuntimeError as e:
                print("  [%s] SKIP wav load: %s" % (stem, e))
                continue

            # Run through golden model
            golden_q  = golden_model.extract(pcm)
            golden_f  = golden_q.astype(np.float32) / (1 << Q_FRAC)

            # Align frame count
            n = min(rtl_mat.shape[1], golden_f.shape[1])
            rtl_al    = rtl_mat[:, :n]
            golden_al = golden_f[:, :n]
            delta     = np.abs(rtl_al - golden_al)

            file_acc = delta_to_accuracy(float(delta.mean()))
            print("  [%s] RTL=%s  golden=%s  mean_delta=%.3f  acc=%.1f%%"
                  % (stem, rtl_mat.shape, golden_f.shape,
                     float(delta.mean()), file_acc))

            kw_rtl_mats.append(rtl_al)
            kw_golden_mats.append(golden_al)

            all_file_rows.append(dict(
                keyword     = keyword,
                stem        = stem,
                n_frames    = n,
                max_delta   = float(delta.max()),
                mean_delta  = float(delta.mean()),
                accuracy_pct = file_acc,
            ))

        if not kw_rtl_mats:
            print("  No files successfully processed for '%s'" % keyword)
            continue

        # Per-keyword averaged spectrogram plot
        plot_path = os.path.join(out_dir, "spectrogram_%s.png" % keyword)
        metrics   = plot_keyword_spectrogram(
            keyword, kw_rtl_mats, kw_golden_mats, plot_path
        )
        kw_summary.append(metrics)

        print("  -> Plot: %s" % plot_path)
        print("  -> Keyword accuracy: %.1f%%  (mean_delta=%.3f log2)"
              % (metrics["accuracy_pct"], metrics["mean_delta"]))

    # -- Full per-file table --------------------------------------------------
    print()
    print("*" * 72)
    print("PER-FILE RESULTS")
    print("*" * 72)
    print("%-12s  %-32s  %9s  %9s  %9s"
          % ("Keyword", "File stem", "Max delt", "Mean delt", "Accuracy"))
    print("-" * 72)
    for r in all_file_rows:
        print("%-12s  %-32s  %9.4f  %9.4f  %8.1f%%"
              % (r["keyword"][:12], r["stem"][:32],
                 r["max_delta"], r["mean_delta"], r["accuracy_pct"]))

    # -- Per-keyword summary table --------------------------------------------
    print()
    print("*" * 72)
    print("PER-KEYWORD ACCURACY SUMMARY")
    print("*" * 72)
    print("%-16s  %6s  %9s  %9s  %10s"
          % ("Keyword", "Files", "Max delt", "Mean delt", "Accuracy"))
    print("-" * 72)
    for r in sorted(kw_summary, key=lambda x: x["accuracy_pct"], reverse=True):
        grade = "EXCELLENT" if r["accuracy_pct"] >= 90 \
                else "GOOD"      if r["accuracy_pct"] >= 70 \
                else "POOR"
        print("%-16s  %6d  %9.4f  %9.4f  %9.1f%%  %s"
              % (r["keyword"][:16], r["n_files"],
                 r["max_delta"], r["mean_delta"],
                 r["accuracy_pct"], grade))

    if kw_summary:
        overall_acc = float(np.mean([r["accuracy_pct"] for r in kw_summary]))
        print("-" * 72)
        print("%-16s  %6s  %9s  %9s  %9.1f%%"
              % ("OVERALL", "", "", "", overall_acc))

    # -- Summary accuracy bar chart -------------------------------------------
    if kw_summary:
        plot_accuracy_summary(
            kw_summary,
            os.path.join(out_dir, "accuracy_summary.png")
        )

    print()
    print("Per-keyword spectrogram plots: spectrogram_<keyword>.png")
    print("Accuracy bar chart:            accuracy_summary.png")


if __name__ == "__main__":
    main()