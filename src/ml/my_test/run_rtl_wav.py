#!/usr/bin/env python3
"""
run_rtl_wav.py — End-to-end RTL-simulated keyword spotting.

Loads a .wav file, runs it through the RTL pipeline (pipeline_top.sv via
cocotb + iverilog), and classifies the resulting log-mel features with DSCNN.

Prerequisites:
    - iverilog (Icarus Verilog) installed
    - cocotb installed (pip install cocotb)
    - numpy, torch, soundfile

Usage:
    python src/ml/my_test/run_rtl_wav.py path/to/audio.wav
    python src/ml/my_test/run_rtl_wav.py audio.wav --seconds 1.0 --start 0.0
    python src/ml/my_test/run_rtl_wav.py audio.wav --model dscnn7-new.pt --topk 5
    python src/ml/my_test/run_rtl_wav.py audio.wav --dump_features feats.npy

Model input shape: (1, 1, n_frames, 40)  — (batch, channel, time, mel_bins)
RTL features:      (40, n_frames) float32, Q4.12 scaled to log2 units
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# ── Paths ─────────────────────────────────────────────────────────────────
_HERE       = Path(__file__).resolve().parent
_SRC_DIR    = _HERE.parent.parent              # src/
_RTL_TOP    = _SRC_DIR / "rtl" / "top"
_DEFAULT_MODEL = _HERE / "dscnn-golden.pt"

# ── Constants (match RTL) ─────────────────────────────────────────────────
SAMPLE_RATE  = 16_000
SAMPLE_W     = 14
SAMPLE_MAX   = (1 << (SAMPLE_W - 1)) - 1  # 8191
N_MELS       = 40
WIN_LEN      = 256
HOP          = 128
STARTUP_LOSS = 3
Q_FRAC       = 12

# Add my_test to path for DSCNN import
sys.path.insert(0, str(_HERE))
from dscnn import DSCNN


# ── Audio loading ─────────────────────────────────────────────────────────

def load_audio(wav_path, start=0.0, seconds=1.0):
    """Load wav, convert to mono 16 kHz, extract segment, return int14 samples."""
    audio, sr = sf.read(str(wav_path), dtype="float32")

    # Stereo -> mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample to 16 kHz if needed
    if sr != SAMPLE_RATE:
        from scipy.signal import resample
        n_out = int(len(audio) * SAMPLE_RATE / sr)
        audio = resample(audio, n_out)
        sr = SAMPLE_RATE

    # Extract segment
    start_sample = int(start * sr)
    end_sample = start_sample + int(seconds * sr)
    audio = audio[start_sample:end_sample]

    # Pad if shorter than requested
    target_len = int(seconds * sr)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))

    # Convert to 14-bit signed integers (matches RTL input width)
    samples = (np.clip(audio, -1.0, 1.0) * SAMPLE_MAX).astype(np.int16)

    print(f"Audio: {wav_path}")
    print(f"  sr={sr}, samples={len(samples)} ({len(samples)/sr:.3f}s)")
    print(f"  range=[{samples.min()}, {samples.max()}] (14-bit: +/-{SAMPLE_MAX})")

    return samples


# ── RTL simulation ────────────────────────────────────────────────────────

def simulate_rtl(samples):
    """Run cocotb simulation with custom samples, return features array."""
    # Save samples for the cocotb test to pick up
    samples_npy = _RTL_TOP / "_rtl_input_samples.npy"
    np.save(str(samples_npy), samples.astype(np.int32))

    features_npy = _RTL_TOP / "rtl_features.npy"

    # Remove stale features file
    if features_npy.exists():
        features_npy.unlink()

    # Expected frame count
    n_samples = len(samples)
    expected_frames = (n_samples - WIN_LEN) // HOP + 1 - STARTUP_LOSS
    print(f"\nRTL simulation:")
    print(f"  samples={n_samples}, expected_frames={expected_frames}")
    print(f"  running cocotb + iverilog... (this may take a minute)")

    # Run make test-cocotb with our custom cocotb module
    env = os.environ.copy()
    env["COCOTB_MODULE"] = "test_wav_pipeline"
    env["SAMPLES_NPY"] = str(samples_npy)

    result = subprocess.run(
        ["make", "test-cocotb"],
        cwd=str(_RTL_TOP),
        env=env,
        capture_output=True,
        text=True,
    )

    # Clean up temp samples
    if samples_npy.exists():
        samples_npy.unlink()

    if result.returncode != 0:
        print("STDERR:", result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
        sys.exit(1)

    # Load features
    if not features_npy.exists():
        print("ERROR: rtl_features.npy was not produced by simulation")
        print("STDOUT (last 1000 chars):", result.stdout[-1000:])
        sys.exit(1)

    features = np.load(str(features_npy))

    # Validate
    n_frames = features.shape[1]
    assert features.shape[0] == N_MELS, f"expected {N_MELS} mels, got {features.shape[0]}"
    assert n_frames > 0, "no frames produced"
    assert not np.any(np.isnan(features)), "NaN in features"
    assert not np.any(np.isinf(features)), "Inf in features"

    if n_frames != expected_frames:
        print(f"  WARNING: got {n_frames} frames, expected {expected_frames}")

    print(f"  OK: {n_frames} frames x {N_MELS} mels")
    print(f"  features: shape={features.shape}, dtype={features.dtype}")
    print(f"  range=[{features.min():.4f}, {features.max():.4f}], "
          f"mean={features.mean():.4f}, std={features.std():.4f}")

    return features


# ── Model loading (reused from classify_rtl.py) ──────────────────────────

def load_model(model_path, device="cpu"):
    """Load DSCNN checkpoint, return (model, labels)."""
    checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)

    config = checkpoint["config"]
    labels = checkpoint.get("labels")
    if labels is None:
        labels = sorted(config.get("data", {}).get("classes", ["silence", "unknown", "yes"]))

    is_quantized = checkpoint.get("quantized", False)

    model_cfg = config.get("model", {})
    pipeline_cfg = checkpoint.get("pipeline") or config.get("pipeline", {})
    feature_cfg = pipeline_cfg.get("feature_extractor", {}) or config.get("preprocessing", {})

    model = DSCNN(
        n_classes=len(labels),
        n_mels=feature_cfg.get("n_mels", 40),
        first_conv_filters=model_cfg.get("first_conv", {}).get("filters", 24),
        first_conv_kernel=tuple(model_cfg.get("first_conv", {}).get("kernel_size", [10, 4])),
        first_conv_stride=tuple(model_cfg.get("first_conv", {}).get("stride", [2, 2])),
        n_ds_blocks=model_cfg.get("ds_blocks", {}).get("n_blocks", 4),
        ds_filters=model_cfg.get("ds_blocks", {}).get("filters", 24),
        ds_kernel=tuple(model_cfg.get("ds_blocks", {}).get("kernel_size", [3, 3])),
        ds_stride=tuple(model_cfg.get("ds_blocks", {}).get("stride", [1, 1])),
    )

    if is_quantized:
        qat_backend = checkpoint.get("qat_backend", "fbgemm")
        model.eval()
        model.fuse_model()
        model.train()
        model.qconfig = torch.quantization.get_default_qat_qconfig(qat_backend)
        torch.quantization.prepare_qat(model, inplace=True)
        model.eval()
        torch.quantization.convert(model, inplace=True)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, labels


# ── Inference ─────────────────────────────────────────────────────────────

def classify(model, features, labels, topk=3):
    """Run DSCNN inference on RTL features, print results."""
    # features: (N_MELS, n_frames) -> model expects (1, 1, n_frames, N_MELS)
    tensor = torch.from_numpy(features.T).unsqueeze(0).unsqueeze(0).float()
    print(f"\nInference:")
    print(f"  input tensor: {tensor.shape}")

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()

    pred_label = labels[pred_idx]

    print(f"\n{'='*50}")
    print(f"  PREDICTION: {pred_label.upper()}")
    print(f"  CONFIDENCE: {confidence*100:.1f}%")
    print(f"{'='*50}")

    print(f"\n  Top-{topk} classes:")
    sorted_indices = np.argsort(probs[0].numpy())[::-1]
    for rank, idx in enumerate(sorted_indices[:topk]):
        label = labels[idx]
        prob = probs[0, idx].item()
        bar_len = int(prob * 30)
        bar = "#" * bar_len + "." * (30 - bar_len)
        marker = " <--" if idx == pred_idx else ""
        print(f"    {rank+1}. {label:>10s}: {prob*100:5.1f}% |{bar}|{marker}")
    print()

    return logits.numpy()


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RTL-simulated keyword spotting: wav -> pipeline_top -> DSCNN",
    )
    parser.add_argument("wav", help="Path to .wav file (16 kHz mono preferred)")
    parser.add_argument("--seconds", type=float, default=1.0, help="Audio duration (default: 1.0)")
    parser.add_argument("--start", type=float, default=0.0, help="Start offset in seconds (default: 0.0)")
    parser.add_argument("--topk", type=int, default=3, help="Top-k classes to show (default: 3)")
    parser.add_argument("--model", "-m", default=str(_DEFAULT_MODEL), help="DSCNN checkpoint path")
    parser.add_argument("--dump_features", default=None, help="Save features to .npy file")
    parser.add_argument("--dump_logits", default=None, help="Save logits to .npy file")

    args = parser.parse_args()

    if not Path(args.wav).exists():
        print(f"ERROR: file not found: {args.wav}")
        sys.exit(1)
    if not Path(args.model).exists():
        print(f"ERROR: model not found: {args.model}")
        sys.exit(1)

    # 1. Load audio
    samples = load_audio(args.wav, start=args.start, seconds=args.seconds)

    # 2. RTL simulation
    features = simulate_rtl(samples)

    # 3. Dump features if requested
    if args.dump_features:
        np.save(args.dump_features, features)
        print(f"  features saved -> {args.dump_features}")

    # 4. Load model
    model, labels = load_model(args.model)
    print(f"Model: {Path(args.model).name}, classes={labels}")

    # 5. Classify
    logits = classify(model, features, labels, topk=args.topk)

    # 6. Dump logits if requested
    if args.dump_logits:
        np.save(args.dump_logits, logits)
        print(f"  logits saved -> {args.dump_logits}")


if __name__ == "__main__":
    main()
