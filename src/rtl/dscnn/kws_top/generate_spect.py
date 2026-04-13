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
import math
import sys
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent                                  # src/rtl/dscnn/kws_top/
REPO_ROOT    = SCRIPT_DIR.parent.parent.parent.parent                 # ML-Audio/
ML_DIR       = REPO_ROOT / "src" / "ml"                              # src/ml/
MODEL_DIR    = REPO_ROOT / "src" / "ml" / "models" / "dscnn"         # src/ml/models/dscnn/
sys.path.insert(0, str(ML_DIR))            # for golden_model.py
sys.path.insert(0, str(MODEL_DIR))         # for dscnn.py

from golden_model import GoldenExtractor
from dscnn import DSCNN

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
DEFAULT_CKPT    = MODEL_DIR / "dscnn-pow2.pt"


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
    if backend == "pow2":
        backend = "qnnpack"
    torch.backends.quantized.engine = backend
    model.eval()
    model.fuse_model()
    model.qconfig = torch.quantization.get_default_qconfig(backend)
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
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET,
                        help="Path to speech_commands dataset root")
    parser.add_argument("--out-dir", type=Path, default=Path("."),
                        help="Output directory for generated files")
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT,
                        help="Path to model checkpoint (.pt)")
    args = parser.parse_args()

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
    global INPUT_SCALE
    INPUT_SCALE = float(model.quant.scale)
    spect_shift = round(-math.log2(INPUT_SCALE))
    spect_shift = max(0, min(15, spect_shift))
    print(f"QuantStub scale: {INPUT_SCALE:.8f}  (SPECT_SHIFT={spect_shift})")

    # Build GoldenExtractor — bit-accurate RTL spectrogram pipeline
    print("Building GoldenExtractor (RTL-accurate spectrogram pipeline)...")
    extractor = GoldenExtractor()

    # Find a WAV file for the requested keyword
    keyword_dir = args.dataset_dir / args.keyword
    if not keyword_dir.exists():
        print(f"ERROR: keyword directory not found: {keyword_dir}", file=sys.stderr)
        sys.exit(1)

    wav_files = sorted(keyword_dir.glob("*.wav"))
    if not wav_files:
        print(f"ERROR: no WAV files in {keyword_dir}", file=sys.stderr)
        sys.exit(1)

    wav_path = wav_files[0]
    print(f"Input WAV      : {wav_path}")

    # Extract spectrogram (50 x 40, matching both training and RTL hardware)
    audio_i16 = load_wav(wav_path)
    spect_float, spect_int8 = extract_spectrogram(audio_i16, extractor)
    print(f"Spectrogram    : {spect_float.shape} float32, range [{spect_float.min():.2f}, {spect_float.max():.2f}]")
    print(f"Quantized INT8 : range [{spect_int8.min()}, {spect_int8.max()}], non-zero={np.count_nonzero(spect_int8)}")

    # Write spectrogram.hex
    spect_hex_path = args.out_dir / "spectrogram.hex"
    write_spectrogram_hex(spect_int8, spect_hex_path)
    print(f"Written        : {spect_hex_path}")

    # Golden inference on the same 50-frame spectrogram the RTL will process
    expected_class = golden_inference(model, spect_float)
    expected_name  = CLASS_NAMES[expected_class]
    print(f"Golden class   : {expected_class} ({expected_name})")
    print(f"Input keyword  : {args.keyword} (class {CLASS_MAP[args.keyword]})")

    if expected_name != args.keyword:
        print(f"WARNING: model predicted '{expected_name}' but input was '{args.keyword}'")
        print(f"         The testbench will check for RTL class == {expected_class} ('{expected_name}')")

    # Write ground_truth_class.txt — the label of the WAV file we actually fed in.
    # This is what the testbench should compare against: did the chip correctly
    # identify the spoken word?  The model prediction (expected_class) may differ
    # from ground truth if the sample is borderline or the model is wrong.
    ground_truth_class = CLASS_MAP[args.keyword]
    ground_truth_name  = args.keyword
    (args.out_dir / "ground_truth_class.txt").write_text(str(ground_truth_class) + "\n")
    (args.out_dir / "ground_truth_name.txt").write_text(ground_truth_name + "\n")

    # Also write the model's own prediction for reference/debugging.
    (args.out_dir / "expected_class.txt").write_text(str(expected_class) + "\n")
    (args.out_dir / "class_name.txt").write_text(expected_name + "\n")
    (args.out_dir / "class_names.txt").write_text("\n".join(CLASS_NAMES) + "\n")
    print(f"Written        : {args.out_dir / 'ground_truth_class.txt'}  ({ground_truth_name})")
    print(f"Written        : {args.out_dir / 'expected_class.txt'}  (model prediction: {expected_name})")
    print(f"Written        : {args.out_dir / 'class_names.txt'}")

    print(f"\nSpectrogram INT8 sample (first 8 values of frame 0):")
    print("  " + "  ".join(f"{int(v):4d}" for v in spect_int8[0, :8]))


if __name__ == "__main__":
    main()
