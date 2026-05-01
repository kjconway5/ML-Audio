#!/usr/bin/env python3
"""
generate_spect_full.py — Test vector generator using the full RTL pipeline golden model.

Identical to generate_spect.py but uses FullPipelineGoldenExtractor (PDM -> CIC ->
compFIR -> STFFT) instead of GoldenExtractor (STFFT only). Use this when testing
a model trained with process_full_data.py.

Usage:
    python3 generate_spect_full.py [--keyword yes] [--dataset-dir /path/to/speech_commands]
                                   [--out-dir spectrograms/] [--ckpt dscnn-32requant-v11/dscnn-32requant-v11.pt]

Outputs (written to --out-dir):
    spectrogram_N.hex  — 2000 signed INT8 values, frame-major (frame*40 + mel)
    test_vectors.json  — manifest with class labels, input scale, per-sample results
    class_names.txt    — human-readable class names

Audio is scaled to 14-bit signed range (±8191) matching the RTL pdm mic adc and process_full_data.py.
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
SCRIPT_DIR  = Path(__file__).parent                           # src/rtl/dscnn/kws_top/
REPO_ROOT   = SCRIPT_DIR.parent.parent.parent.parent          # ML-Audio/
ML_DIR      = REPO_ROOT / "src" / "ml"                       # src/ml/
MODELS_DIR  = REPO_ROOT / "src" / "ml" / "models"            # src/ml/models/
PIPELINE_DIR = REPO_ROOT / "src" / "ml" / "Pipeline"         # src/ml/Pipeline/  (dscnn.py lives here)
sys.path.insert(0, str(ML_DIR))          # for golden_model.py
sys.path.insert(0, str(PIPELINE_DIR))   # for dscnn.py
sys.path.insert(0, str(SCRIPT_DIR))     # for rtl_golden.py

from golden_model import FullPipelineGoldenExtractor
from dscnn import DSCNN
from rtl_golden import rtl_golden_predict, load_layer_cfgs, load_hex_file as rtl_load_hex

# ── Constants ─────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
N_MELS      = 40
N_FRAMES    = 50
SPECT_DEPTH = N_FRAMES * N_MELS   # 2000

# 14-bit ADC range — matches RTL hardware and process_full_data.py
SAMPLE_W   = 14
SAMPLE_MAX = (1 << (SAMPLE_W - 1)) - 1   # 8191

INPUT_SCALE  = None   # set in main() from model.quant.scale
CLASS_NAMES  = None   # set in main() from checkpoint
CLASS_MAP    = None   # set in main() from checkpoint

DEFAULT_DATASET = ML_DIR / "Pipeline" / "speech_commands_v0.02"


# ── Audio loading ─────────────────────────────────────────────────────────────

def load_wav(path: Path) -> np.ndarray:
    """Load WAV, return 14-bit signed int16 mono at 16 kHz."""
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz, got {sr} Hz: {path}")
    return np.clip(audio * SAMPLE_MAX, -SAMPLE_MAX - 1, SAMPLE_MAX).astype(np.int16)


# ── Spectrogram extraction ────────────────────────────────────────────────────

def extract_spectrogram(audio_i14: np.ndarray,
                        extractor: FullPipelineGoldenExtractor) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the full PDM -> CIC -> compFIR -> STFFT pipeline.
    Returns (spect_float, spect_int8), each shape (N_FRAMES, N_MELS), frame-major.
    """
    mel_float = extractor.extract_float(audio_i14).T   # (n_frames, N_MELS)

    n = mel_float.shape[0]
    if n >= N_FRAMES:
        start = (n - N_FRAMES) // 2
        mel_float = mel_float[start:start + N_FRAMES, :]
    else:
        pad = N_FRAMES - n
        mel_float = np.pad(mel_float, ((0, pad), (0, 0)), mode="constant")

    mel_int8 = np.round(mel_float / INPUT_SCALE).clip(-128, 127).astype(np.int8)
    return mel_float, mel_int8


# ── Model loading / inference ─────────────────────────────────────────────────

def _int8_sym_qconfig():
    obs = torch.quantization.observer.MinMaxObserver.with_args(
        dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, reduce_range=False,
    )
    return torch.quantization.QConfig(activation=obs, weight=obs)


def load_model(ckpt_path: Path) -> torch.nn.Module:
    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg    = checkpoint["config"]["model"]
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

    backend  = checkpoint.get("qat_backend", "fbgemm")
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
    x = torch.from_numpy(spect_float).float().unsqueeze(0).unsqueeze(0)  # (1,1,50,40)
    with torch.no_grad():
        logits = model(x)
    return int(logits.argmax(dim=1).item())


# ── Output writers ────────────────────────────────────────────────────────────

def write_spectrogram_hex(spect_int8: np.ndarray, path: Path):
    flat = spect_int8.flatten()
    assert len(flat) == SPECT_DEPTH
    with open(path, "w") as f:
        for v in flat:
            f.write(f"{int(v) & 0xFF:02x}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate RTL test vectors using full pipeline (PDM->CIC->FIR->STFFT)")
    parser.add_argument("--keyword",     default="yes")
    parser.add_argument("--n-samples",   type=int, default=10)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir",     type=Path, default=Path("spectrograms"))
    parser.add_argument("--ckpt",        type=Path, required=True,
                        help="Checkpoint relative to src/ml/models/ or absolute path")
    parser.add_argument("--wav-file",    type=Path, default=None)
    parser.add_argument("--seed",        type=int,  default=None)
    args = parser.parse_args()

    if not args.ckpt.is_absolute():
        args.ckpt = MODELS_DIR / args.ckpt
    if not args.ckpt.exists():
        # fallback: try relative to MODELS_DIR by name only
        alt = MODELS_DIR / args.ckpt.name
        if alt.exists():
            args.ckpt = alt
        else:
            print(f"ERROR: checkpoint not found: {args.ckpt}", file=sys.stderr)
            sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model      : {args.ckpt}")
    model = load_model(args.ckpt)
    ckpt  = torch.load(str(args.ckpt), map_location="cpu", weights_only=False)

    global CLASS_NAMES, CLASS_MAP
    CLASS_NAMES = ckpt["labels"]
    CLASS_MAP   = ckpt["label_to_id"]
    print(f"Class labels       : {CLASS_NAMES}")

    if args.keyword not in CLASS_MAP:
        print(f"ERROR: '{args.keyword}' not in labels {CLASS_NAMES}", file=sys.stderr)
        sys.exit(1)

    global INPUT_SCALE
    INPUT_SCALE = float(model.quant.scale)
    Q_FRAC_LOG  = 10
    spect_shift = Q_FRAC_LOG - round(-math.log2(INPUT_SCALE))
    spect_shift = max(0, min(15, spect_shift))
    print(f"QuantStub scale    : {INPUT_SCALE:.8f}  (SPECT_SHIFT={spect_shift})")
    print(f"Audio scaling      : float32 [-1,1] -> int14 [±{SAMPLE_MAX}]")

    print("Building FullPipelineGoldenExtractor (PDM -> CIC -> compFIR -> STFFT)...")
    extractor = FullPipelineGoldenExtractor()

    n_filters = ckpt["config"]["model"]["ds_blocks"]["filters"]
    scales_path  = args.ckpt.parent / "scales.txt"
    weights_path = args.ckpt.parent / "weights.hex"
    bias_path    = args.ckpt.parent / "bias.hex"
    rtl_available = all(p.exists() for p in [scales_path, weights_path, bias_path])
    if rtl_available:
        layer_cfgs  = load_layer_cfgs(scales_path, n_filters=n_filters)
        rtl_weights = rtl_load_hex(str(weights_path), signed=True, width=8)
        rtl_biases  = rtl_load_hex(str(bias_path),   signed=True, width=32)
        print(f"RTL arithmetic     : {len(rtl_weights)} weights, {len(rtl_biases)} biases")
    else:
        print("WARNING: weights.hex/bias.hex/scales.txt not found — run export.py first")

    # ── Collect WAV files ─────────────────────────────────────────────────────
    if args.wav_file is not None:
        if not args.wav_file.exists():
            print(f"ERROR: WAV file not found: {args.wav_file}", file=sys.stderr)
            sys.exit(1)
        wav_list = [args.wav_file]
    elif args.keyword == "silence":
        silence_dir = args.dataset_dir / "_generated_silence_"
        if not silence_dir.exists():
            print(f"ERROR: silence directory not found: {silence_dir}", file=sys.stderr)
            sys.exit(1)
        wav_files = sorted(silence_dir.glob("*.wav"))
        rng = random.Random(args.seed)
        wav_list = sorted(rng.sample(wav_files, min(args.n_samples, len(wav_files))))
    elif args.keyword == "unknown":
        target_keywords = {k for k in CLASS_NAMES if k not in ("silence", "unknown")}
        unknown_dirs = [
            d for d in sorted(args.dataset_dir.iterdir())
            if d.is_dir() and not d.name.startswith("_") and d.name not in target_keywords
        ]
        wav_files = sorted(p for d in unknown_dirs for p in d.glob("*.wav"))
        rng = random.Random(args.seed)
        wav_list = sorted(rng.sample(wav_files, min(args.n_samples, len(wav_files))))
    else:
        keyword_dir = args.dataset_dir / args.keyword
        if not keyword_dir.exists():
            print(f"ERROR: keyword directory not found: {keyword_dir}", file=sys.stderr)
            sys.exit(1)
        wav_files = sorted(keyword_dir.glob("*.wav"))
        rng = random.Random(args.seed)
        wav_list = sorted(rng.sample(wav_files, min(args.n_samples, len(wav_files))))

    ground_truth_class = CLASS_MAP[args.keyword]

    # ── Process each WAV ──────────────────────────────────────────────────────
    out_lines = []
    out_lines.append(f"\nGenerating {len(wav_list)} test vector(s) for keyword '{args.keyword}'  "
                     f"[full pipeline: PDM->CIC->FIR->STFFT]")
    out_lines.append(f"{'#':<4}  {'wav':<35}  {'nz':>5}  {'pytorch':>8}  {'arith':>8}  {'match':>6}")
    out_lines.append("-" * 80)

    samples = []
    for i, wav_path in enumerate(wav_list):
        audio_i14 = load_wav(wav_path)
        spect_float, spect_int8 = extract_spectrogram(audio_i14, extractor)

        hex_filename = f"spectrogram_{i}.hex"
        write_spectrogram_hex(spect_int8, args.out_dir / hex_filename)
        hex_rel = str((args.out_dir / hex_filename).resolve().relative_to(SCRIPT_DIR.resolve()))

        pytorch_class = golden_inference(model, spect_float)
        pytorch_name  = CLASS_NAMES[pytorch_class]

        if rtl_available:
            spect_list = [int(v) for v in spect_int8.flatten()]
            arith_class, arith_gap = rtl_golden_predict(spect_list, rtl_weights, rtl_biases, layer_cfgs)
            arith_name  = CLASS_NAMES[arith_class]
            sorted_gap  = sorted(enumerate(arith_gap), key=lambda x: x[1], reverse=True)
            margin      = sorted_gap[0][1] - sorted_gap[1][1]
            gap_str     = "  ".join(f"{CLASS_NAMES[c]}:{v}" for c, v in sorted_gap)
        else:
            arith_class, arith_name, margin, gap_str = None, "N/A", 0, "N/A"

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
            "index":              i,
            "hex_file":           hex_rel,
            "wav":                str(wav_path),
            "ground_truth_class": ground_truth_class,
            "ground_truth_name":  args.keyword,
            "pytorch_class":      pytorch_class,
            "pytorch_name":       pytorch_name,
            "arith_class":        arith_class,
            "arith_name":         arith_name,
            "non_zero":           int(np.count_nonzero(spect_int8)),
        })

    # ── Write manifest ────────────────────────────────────────────────────────
    manifest = {
        "keyword":            args.keyword,
        "ground_truth_class": ground_truth_class,
        "class_names":        CLASS_NAMES,
        "input_scale":        INPUT_SCALE,
        "spect_shift":        spect_shift,
        "pipeline":           "full",
        "audio_scaling":      f"int14 [±{SAMPLE_MAX}]",
        "seed":               args.seed,
        "samples":            samples,
    }
    manifest_path = args.out_dir / "test_vectors.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    (args.out_dir / "class_names.txt").write_text("\n".join(CLASS_NAMES) + "\n")

    correct = sum(1 for s in samples if
                  (s["arith_name"] if rtl_available else s["pytorch_name"]) == args.keyword)
    out_lines.append(f"\nGolden model accuracy: {correct}/{len(samples)}")
    out_lines.append(f"Written: {manifest_path}")
    out_lines.append(f"Written: {len(samples)} spectrogram hex file(s) in {args.out_dir}")

    output = "\n".join(out_lines)
    print(output)

    from datetime import datetime
    results_path = args.ckpt.parent / "gen_spect_results.txt"
    seed_str = str(args.seed) if args.seed is not None else "random"
    with open(results_path, "a") as rf:
        rf.write(f"=== {args.keyword}  ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})  "
                 f"seed={seed_str}  pipeline=full ===\n")
        rf.write(output + "\n\n")


if __name__ == "__main__":
    main()
