#!/usr/bin/env python3
"""
generate_spect.py — Test vector generator for kws_top end-to-end RTL testbench.

Usage:
    python3 generate_spect.py [--keyword yes] [--dataset-dir /path/to/speech_commands]
                              [--out-dir /path/to/output] [--ckpt /path/to/model.pt]

Outputs (written to --out-dir, default current directory):
    spectrogram.hex    — 2000 signed INT8 values (one per line, 2-digit hex),
                         stored frame-major: addr = frame*40 + mel
    expected_class.txt — integer class index (0-6)
    class_name.txt     — human-readable class name

Class map (alphabetical, from checkpoint):
    no=0  off=1  on=2  silence=3  unknown=4  wow=5  yes=6

Preprocessing uses GoldenExtractor (golden_model.py) — bit-accurate replica of the
RTL logmel_top pipeline (fixed-point Hanning window, 18-bit FFT, Q0.15 mel filterbank,
log2 LUT). This matches what the RTL will compute in hardware.

Features are cropped/padded to exactly N_FRAMES x N_MELS = 50 x 40, matching the
FSM's fixed ifmap_h=50, ifmap_w=40. The model is also trained on 50-frame inputs.

Quantization to INT8 uses model.quant.scale (QuantStub's learned input scale).
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent                                  # src/rtl/dscnn/kws_top/
REPO_ROOT    = SCRIPT_DIR.parent.parent.parent.parent                 # ML-Audio/
ML_DIR       = REPO_ROOT / "src" / "ml"                              # src/ml/
MODELS_DIR   = REPO_ROOT / "src" / "ml" / "models"                   # src/ml/models/  (base for --ckpt)
DSCNN_DIR    = REPO_ROOT / "src" / "ml" / "models" / "dscnn"         # src/ml/models/dscnn/ (dscnn.py lives here)
sys.path.insert(0, str(ML_DIR))            # for golden_model.py
sys.path.insert(0, str(DSCNN_DIR))         # for dscnn.py
sys.path.insert(0, str(SCRIPT_DIR))        # for rtl_golden.py

from golden_model import GoldenExtractor
from dscnn import DSCNN
from rtl_golden import rtl_golden_predict, load_layer_cfgs, load_hex_file as rtl_load_hex

# ── Constants (must match FSM hardcoded values and training config) ────────────
SAMPLE_RATE  = 16000
N_MELS       = 40
N_FRAMES     = 50    # FSM hardcodes ifmap_h=50, ifmap_w=40
SPECT_DEPTH  = N_FRAMES * N_MELS  # 2000

# INPUT_SCALE is loaded at runtime from model.quant.scale (the QuantStub output scale).
INPUT_SCALE  = None   # set in main() after model load

# CLASS_NAMES and CLASS_MAP are loaded from the checkpoint at runtime.
CLASS_NAMES  = None   # set in main() from checkpoint
CLASS_MAP    = None   # set in main() from checkpoint

DEFAULT_DATASET = ML_DIR / "Pipeline" / "speech_commands_v0.02"


# ── Audio loading ─────────────────────────────────────────────────────────────

def load_wav(path: Path) -> np.ndarray:
    """Load a WAV file and return int16-range int16 mono at 16 kHz."""
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz, got {sr} Hz: {path}")
    # Scale normalized float [-1,1] to int16 range, matching process_data.py
    return np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)


# ── Spectrogram extraction ────────────────────────────────────────────────────

def extract_spectrogram(audio_i16: np.ndarray, extractor: GoldenExtractor) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract log2-mel spectrogram using GoldenExtractor (bit-accurate RTL replica).

    Args:
        audio_i16   — int16 mono audio at 16 kHz
        extractor   — GoldenExtractor instance

    Returns:
        spect_float — float32 log2-mel, shape (N_FRAMES, N_MELS), frame-major
        spect_int8  — INT8 quantized, shape (N_FRAMES, N_MELS), frame-major
    """
    # extract_float returns (N_MELS, n_frames) float32 log2 values
    mel_float = extractor.extract_float(audio_i16).T  # (n_frames, N_MELS)

    # Center-crop to exactly N_FRAMES — must match process_data.py center-crop logic
    n = mel_float.shape[0]
    if n >= N_FRAMES:
        start = (n - N_FRAMES) // 2
        mel_float = mel_float[start:start + N_FRAMES, :]
    else:
        pad = N_FRAMES - n
        mel_float = np.pad(mel_float, ((0, pad), (0, 0)), mode="constant")

    # Quantize to INT8 using QuantStub input scale (matches QAT training)
    mel_int8 = np.round(mel_float / INPUT_SCALE).clip(-128, 127).astype(np.int8)

    return mel_float, mel_int8


# ── PyTorch golden inference ──────────────────────────────────────────────────

def _int8_sym_qconfig():
    obs = torch.quantization.observer.MinMaxObserver.with_args(
        dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, reduce_range=False,
    )
    return torch.quantization.QConfig(activation=obs, weight=obs)


def load_model(ckpt_path: Path) -> torch.nn.Module:
    """Load and reconstruct the QAT-converted INT8 DSCNN from checkpoint."""
    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]["model"]
    preproc = checkpoint["config"]["preprocessing"]

    model = DSCNN(
        n_classes=cfg["n_classes"],
        n_mels=preproc["n_mels"],
        first_conv_filters=cfg["first_conv"]["filters"],
        first_conv_kernel=tuple(cfg["first_conv"]["kernel_size"]),
        first_conv_stride=tuple(cfg["first_conv"]["stride"]),
        n_ds_blocks=cfg["ds_blocks"]["n_blocks"],
        ds_filters=cfg["ds_blocks"]["filters"],
        ds_kernel=tuple(cfg["ds_blocks"]["kernel_size"]),
        ds_stride=tuple(cfg["ds_blocks"]["stride"]),
    )

    # Reconstruct the quantized model: fuse → prepare → convert → load weights
    backend = checkpoint.get("qat_backend", "fbgemm")
    int8_sym = (backend == "qnnpack_int8sym")
    if backend in ("pow2", "qnnpack_int8sym"):
        backend = "qnnpack"
    torch.backends.quantized.engine = backend
    model.eval()
    model.fuse_model()
    model.qconfig = _int8_sym_qconfig() if int8_sym else torch.quantization.get_default_qconfig(backend)
    torch.quantization.prepare(model, inplace=True)
    torch.quantization.convert(model, inplace=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def golden_inference(model: torch.nn.Module, spect_float: np.ndarray) -> int:
    """
    Run the full PyTorch model on the 50-frame float spectrogram.
    Input shape expected by model: (1, 1, N_FRAMES, N_MELS) = (1, 1, 50, 40).
    This matches both the training input shape and the RTL hardware input.
    """
    x = torch.from_numpy(spect_float).float()   # (50, 40)
    x = x.unsqueeze(0).unsqueeze(0)             # (1, 1, 50, 40)
    with torch.no_grad():
        logits = model(x)                        # (1, 7)
    return int(logits.argmax(dim=1).item())


# ── Output writers ────────────────────────────────────────────────────────────

def write_spectrogram_hex(spect_int8: np.ndarray, path: Path):
    """Write 2000 INT8 values, one per line, as 2-digit hex (two's complement)."""
    flat = spect_int8.flatten()
    assert len(flat) == SPECT_DEPTH, f"Expected {SPECT_DEPTH} values, got {len(flat)}"
    with open(path, "w") as f:
        for v in flat:
            f.write(f"{int(v) & 0xFF:02x}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate RTL test vectors for kws_top")
    parser.add_argument("--keyword", default="yes",
                        help="Keyword to use (default: yes)")
    parser.add_argument("--n-samples", type=int, default=10,
                        help="Number of WAV samples to generate test vectors for (default: 10)")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET,
                        help="Path to speech_commands dataset root")
    parser.add_argument("--out-dir", type=Path, default=Path("spectrograms"),
                        help="Output directory for generated files (default: spectrograms/)")
    parser.add_argument("--ckpt", type=Path, required=True,
                        help="Model checkpoint relative to src/ml/models/  "
                             "(e.g. dscnn-pow2-v7/dscnn-pow2-v7.pt) "
                             "or an absolute path")
    parser.add_argument("--wav-file", type=Path, default=None,
                        help="Use a specific WAV file (overrides --n-samples, generates 1 sample)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible file selection (default: random)")
    args = parser.parse_args()

    # Resolve --ckpt: if not absolute, treat as relative to src/ml/models/
    if not args.ckpt.is_absolute():
        args.ckpt = MODELS_DIR / args.ckpt
    if not args.ckpt.exists():
        print(f"ERROR: checkpoint not found: {args.ckpt}", file=sys.stderr)
        print(f"       (looked in {MODELS_DIR} for relative paths)", file=sys.stderr)
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load model — derives CLASS_NAMES, INPUT_SCALE, provides golden inference
    print(f"Loading model  : {args.ckpt}")
    model = load_model(args.ckpt)
    ckpt  = torch.load(str(args.ckpt), map_location="cpu", weights_only=False)

    global CLASS_NAMES, CLASS_MAP
    CLASS_NAMES = ckpt["labels"]
    CLASS_MAP   = ckpt["label_to_id"]
    print(f"Class labels   : {CLASS_NAMES}")

    if args.keyword not in CLASS_MAP:
        print(f"ERROR: keyword '{args.keyword}' not in checkpoint labels {CLASS_NAMES}",
              file=sys.stderr)
        sys.exit(1)

    # Derive INPUT_SCALE from the QuantStub (model.quant.scale)
    # RTL: int8 = Q6.10_uint16 >> SPECT_SHIFT → match requires SPECT_SHIFT = Q_FRAC - round(-log2(scale))
    global INPUT_SCALE
    INPUT_SCALE = float(model.quant.scale)
    Q_FRAC_LOG = 10  # must match golden_model.py Q_FRAC
    spect_shift = Q_FRAC_LOG - round(-math.log2(INPUT_SCALE))
    spect_shift = max(0, min(15, spect_shift))
    print(f"QuantStub scale: {INPUT_SCALE:.8f}  (SPECT_SHIFT={spect_shift})")

    # Build GoldenExtractor — bit-accurate RTL spectrogram pipeline
    print("Building GoldenExtractor (RTL-accurate spectrogram pipeline)...")
    extractor = GoldenExtractor()

    # Detect filter count from checkpoint config so the correct architecture table is used
    n_filters = ckpt["config"]["model"]["ds_blocks"]["filters"]

    # Load RTL arithmetic golden (weights, biases, layer configs from scales.txt)
    scales_path  = args.ckpt.parent / "scales.txt"
    weights_path = args.ckpt.parent / "weights.hex"
    bias_path    = args.ckpt.parent / "bias.hex"
    rtl_available = all(p.exists() for p in [scales_path, weights_path, bias_path])
    if rtl_available:
        layer_cfgs  = load_layer_cfgs(scales_path, n_filters=n_filters)
        rtl_weights = rtl_load_hex(str(weights_path), signed=True, width=8)
        rtl_biases  = rtl_load_hex(str(bias_path),   signed=True, width=32)
        print(f"RTL arithmetic : loaded ({len(rtl_weights)} weights, {len(rtl_biases)} biases, {n_filters} filters)")
    else:
        print("WARNING: weights.hex/bias.hex/scales.txt not found — RTL arithmetic column unavailable")
        print("         Run export.py first to generate these files")

    # Collect WAV files to process
    if args.wav_file is not None:
        if not args.wav_file.exists():
            print(f"ERROR: WAV file not found: {args.wav_file}", file=sys.stderr)
            sys.exit(1)
        wav_list = [args.wav_file]
    elif args.keyword == "silence":
        # Silence has no named directory — use _generated_silence_/
        silence_dir = args.dataset_dir / "_generated_silence_"
        if not silence_dir.exists():
            print(f"ERROR: silence directory not found: {silence_dir}", file=sys.stderr)
            sys.exit(1)
        wav_files = sorted(silence_dir.glob("*.wav"))
        if not wav_files:
            print(f"ERROR: no WAV files in {silence_dir}", file=sys.stderr)
            sys.exit(1)
        rng = random.Random(args.seed)
        n = min(args.n_samples, len(wav_files))
        wav_list = sorted(rng.sample(wav_files, n))
    elif args.keyword == "unknown":
        # Unknown is assembled from all non-target, non-special word directories
        target_keywords = {k for k in CLASS_NAMES if k not in ("silence", "unknown")}
        unknown_dirs = [
            d for d in sorted(args.dataset_dir.iterdir())
            if d.is_dir() and not d.name.startswith("_") and d.name not in target_keywords
        ]
        if not unknown_dirs:
            print(f"ERROR: no unknown word directories found in {args.dataset_dir}", file=sys.stderr)
            sys.exit(1)
        wav_files = sorted(p for d in unknown_dirs for p in d.glob("*.wav"))
        if not wav_files:
            print(f"ERROR: no WAV files found in unknown directories", file=sys.stderr)
            sys.exit(1)
        rng = random.Random(args.seed)
        n = min(args.n_samples, len(wav_files))
        wav_list = sorted(rng.sample(wav_files, n))
    else:
        keyword_dir = args.dataset_dir / args.keyword
        if not keyword_dir.exists():
            print(f"ERROR: keyword directory not found: {keyword_dir}", file=sys.stderr)
            sys.exit(1)
        wav_files = sorted(keyword_dir.glob("*.wav"))
        if not wav_files:
            print(f"ERROR: no WAV files in {keyword_dir}", file=sys.stderr)
            sys.exit(1)
        rng = random.Random(args.seed)
        n = min(args.n_samples, len(wav_files))
        wav_list = sorted(rng.sample(wav_files, n))

    ground_truth_class = CLASS_MAP[args.keyword]

    # ── Process each WAV and write numbered spectrogram files ─────────────────
    # Collect table output so it can be printed to stdout and saved to the results file.
    out_lines = []
    out_lines.append(f"\nGenerating {len(wav_list)} test vector(s) for keyword '{args.keyword}'")
    out_lines.append(f"{'#':<4}  {'wav':<35}  {'nz':>5}  {'pytorch':>8}  {'arith':>8}  {'match':>6}")
    out_lines.append("-" * 80)

    samples = []
    for i, wav_path in enumerate(wav_list):
        audio_i16 = load_wav(wav_path)
        spect_float, spect_int8 = extract_spectrogram(audio_i16, extractor)

        # Write spectrogram_{i}.hex
        hex_filename = f"spectrogram_{i}.hex"
        write_spectrogram_hex(spect_int8, args.out_dir / hex_filename)
        # Store path relative to SCRIPT_DIR (kws_top/) so test_kws_top.py can resolve it
        hex_filename = str((args.out_dir / hex_filename).resolve().relative_to(SCRIPT_DIR.resolve()))

        # PyTorch model prediction (float QAT model)
        pytorch_class = golden_inference(model, spect_float)
        pytorch_name  = CLASS_NAMES[pytorch_class]

        # RTL integer arithmetic prediction (canonical — matches what RTL chip does)
        if rtl_available:
            spect_list = [int(v) for v in spect_int8.flatten()]
            arith_class, arith_gap = rtl_golden_predict(spect_list, rtl_weights, rtl_biases, layer_cfgs)
            arith_name = CLASS_NAMES[arith_class]
            sorted_gap = sorted(enumerate(arith_gap), key=lambda x: x[1], reverse=True)
            margin     = sorted_gap[0][1] - sorted_gap[1][1]
            gap_str    = "  ".join(f"{CLASS_NAMES[c]}:{v}" for c, v in sorted_gap)
        else:
            arith_class, arith_name, margin, gap_str = None, "N/A", 0, "N/A"

        # match is based on RTL arithmetic (the true ground truth for RTL verification)
        if rtl_available:
            match = "OK" if arith_name == args.keyword else "MISS"
            if pytorch_name != arith_name:
                match += " (pytorch≠arith)"
        else:
            match = "OK" if pytorch_name == args.keyword else "MISS"

        out_lines.append(f"{i:<4}  {wav_path.name:<35}  {np.count_nonzero(spect_int8):>5}  "
                         f"{pytorch_name:>8}  {arith_name:>8}  {match}")
        if rtl_available:
            out_lines.append(f"      GAP: {gap_str}  margin={margin}")

        samples.append({
            "index":               i,
            "hex_file":            hex_filename,
            "wav":                 str(wav_path),
            "ground_truth_class":  ground_truth_class,
            "ground_truth_name":   args.keyword,
            "pytorch_class":       pytorch_class,
            "pytorch_name":        pytorch_name,
            "arith_class":         arith_class,
            "arith_name":          arith_name,
            "non_zero":            int(np.count_nonzero(spect_int8)),
        })

    # ── Write manifest ────────────────────────────────────────────────────────
    manifest = {
        "keyword":            args.keyword,
        "ground_truth_class": ground_truth_class,
        "class_names":        CLASS_NAMES,
        "input_scale":        INPUT_SCALE,
        "spect_shift":        spect_shift,
        "seed":               args.seed,
        "samples":            samples,
    }
    manifest_path = args.out_dir / "test_vectors.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Write class_names.txt for testbench backward compat
    (args.out_dir / "class_names.txt").write_text("\n".join(CLASS_NAMES) + "\n")

    # When RTL arithmetic files are unavailable, fall back to pytorch prediction for scoring
    correct = sum(1 for s in samples if
                  (s["arith_name"] if rtl_available else s["pytorch_name"]) == args.keyword)
    out_lines.append(f"\nGolden model accuracy on these samples: {correct}/{len(samples)}")
    out_lines.append(f"Written: {manifest_path}")
    out_lines.append(f"Written: {len(samples)} spectrogram hex file(s) in {args.out_dir}")

    # ── Print to stdout and append to per-model results file ─────────────────
    output = "\n".join(out_lines)
    print(output)

    from datetime import datetime
    results_path = args.ckpt.parent / "gen_spect_results.txt"
    seed_str = str(args.seed) if args.seed is not None else "random"
    with open(results_path, "a") as rf:
        rf.write(f"=== {args.keyword}  ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})  seed={seed_str} ===\n")
        rf.write(output + "\n\n")


if __name__ == "__main__":
    main()
