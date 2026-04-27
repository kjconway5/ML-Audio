#!/usr/bin/env python3
"""
diagnosis_full.py — Definitive diagnosis of training/inference feature mismatch.

Proves that dscnn-golden.pt was trained on torchaudio LogMelExtractor features
(int16 input, range ~20-40 log2) but inference uses GoldenExtractor features
(int14 input, range 0-16 Q4.12).

Usage:
    python src/ml/my_test/diagnosis_full.py
    python src/ml/my_test/diagnosis_full.py --model src/ml/my_test/dscnn-golden.pt
    python src/ml/my_test/diagnosis_full.py --wav path/to/test.wav
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "Pipeline"))

from dscnn import DSCNN
from golden_model import GoldenExtractor
from features import LogMelExtractor


def load_quantized_model(model_path):
    """Load DSCNN checkpoint with correct quantization setup."""
    ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
    labels = ckpt["labels"]
    config = ckpt["config"]
    model_cfg = config.get("model", {})

    model = DSCNN(
        n_classes=len(labels),
        n_mels=config.get("preprocessing", {}).get("n_mels", 40),
        first_conv_filters=model_cfg.get("first_conv", {}).get("filters", 24),
        first_conv_kernel=tuple(model_cfg.get("first_conv", {}).get("kernel_size", [10, 4])),
        first_conv_stride=tuple(model_cfg.get("first_conv", {}).get("stride", [2, 2])),
        n_ds_blocks=model_cfg.get("ds_blocks", {}).get("n_blocks", 4),
        ds_filters=model_cfg.get("ds_blocks", {}).get("filters", 24),
        ds_kernel=tuple(model_cfg.get("ds_blocks", {}).get("kernel_size", [3, 3])),
        ds_stride=tuple(model_cfg.get("ds_blocks", {}).get("stride", [1, 1])),
    )

    if ckpt.get("quantized", False):
        backend = ckpt.get("qat_backend", "fbgemm")
        model.eval()
        model.fuse_model()
        model.train()
        model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
        torch.quantization.prepare_qat(model, inplace=True)
        model.eval()
        torch.quantization.convert(model, inplace=True)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, labels, ckpt


def infer(model, features_2d, labels):
    """Run inference on (n_mels, n_frames) features. Returns (pred_label, confidence, probs)."""
    tensor = torch.from_numpy(features_2d.T).unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].numpy()
    idx = np.argmax(probs)
    return labels[idx], probs[idx], probs, tensor.shape


def print_probs(labels, probs, prefix=""):
    for i, l in enumerate(labels):
        print(f"{prefix}  {l:>10s}: {probs[i]*100:5.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(_HERE / "dscnn-golden.pt"))
    parser.add_argument("--wav", default=None, help="Optional WAV file to test")
    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────────────────
    model, labels, ckpt = load_quantized_model(args.model)
    sd = ckpt["model_state_dict"]
    q_scale = sd["quant.scale"].item()
    q_zp = sd["quant.zero_point"].item()

    print("=" * 70)
    print("  TRAINING vs INFERENCE FEATURE MISMATCH DIAGNOSIS")
    print("=" * 70)
    print(f"\nModel: {Path(args.model).name}")
    print(f"Labels: {labels}")
    print(f"Quantized: {ckpt.get('quantized')}, backend: {ckpt.get('qat_backend')}")
    print(f"Test accuracy (on training data): {ckpt.get('test_accuracy', 'N/A')}")

    # ── QuantStub analysis ────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"STEP 1: QuantStub Observer Analysis")
    print(f"{'─'*70}")
    obs_max = q_scale * 255
    print(f"  quant.scale = {q_scale:.6f}")
    print(f"  quant.zero_point = {q_zp}")
    print(f"  Implied input range (quint8 0-255): [0, {obs_max:.2f}]")
    print(f"  Expected range if trained on golden:     [0, ~16]  (scale ~0.063)")
    print(f"  Expected range if trained on torchaudio:  [0, ~35]  (scale ~0.137)")

    if obs_max > 20:
        print(f"\n  ** FINDING: Observer range {obs_max:.1f} >> 16 (golden max).")
        print(f"  ** Model was trained on TORCHAUDIO features, NOT golden. **")
    else:
        print(f"\n  Observer range is consistent with golden features.")

    # ── Generate test audio ───────────────────────────────────────────────
    if args.wav:
        import soundfile as sf
        audio_f, sr = sf.read(args.wav, dtype="float32")
        if audio_f.ndim > 1:
            audio_f = audio_f.mean(axis=1)
        if sr != 16000:
            from scipy.signal import resample
            audio_f = resample(audio_f, int(len(audio_f) * 16000 / sr))
        if len(audio_f) < 16000:
            audio_f = np.pad(audio_f, (0, 16000 - len(audio_f)))
        else:
            audio_f = audio_f[:16000]
        audio_f = np.clip(audio_f, -1.0, 1.0)
        source = f"WAV file: {args.wav}"
    else:
        np.random.seed(42)
        t = np.arange(16000) / 16000.0
        audio_f = (0.3 * np.sin(2 * np.pi * 300 * t)
                   + 0.15 * np.sin(2 * np.pi * 800 * t)
                   + 0.05 * np.random.randn(16000))
        audio_f = np.clip(audio_f, -1.0, 1.0)
        source = "Synthetic 300+800 Hz"

    audio_i16 = (audio_f * 32767).astype(np.int16)
    audio_i14 = (audio_f * 8191).astype(np.int16)

    print(f"\n{'─'*70}")
    print(f"STEP 2: Feature Extraction Comparison")
    print(f"{'─'*70}")
    print(f"  Source: {source}")
    print(f"  Samples: {len(audio_f)}")
    print(f"  Audio RMS: {np.sqrt(np.mean(audio_f**2)):.4f}")

    # Torchaudio extractor (training path)
    le = LogMelExtractor(sample_rate=16000, n_fft=256, n_mels=40,
                         hop_length=128, window_length=256)
    torch_feats, _ = le.extract(audio_i16)  # (40, n_frames)

    # Golden extractor (inference path)
    ge = GoldenExtractor()
    golden_feats = ge.extract_float(audio_i14)  # (40, n_frames)

    print(f"\n  [TORCHAUDIO — training pipeline (process_data.py)]")
    print(f"    Input: int16 ±32767")
    print(f"    Shape: {torch_feats.shape}")
    print(f"    Range: [{torch_feats.min():.2f}, {torch_feats.max():.2f}]")
    print(f"    Mean:  {torch_feats.mean():.2f}, Std: {torch_feats.std():.2f}")

    print(f"\n  [GOLDEN — inference pipeline (golden_model.py)]")
    print(f"    Input: int14 ±8191")
    print(f"    Shape: {golden_feats.shape}")
    print(f"    Range: [{golden_feats.min():.2f}, {golden_feats.max():.2f}]")
    print(f"    Mean:  {golden_feats.mean():.2f}, Std: {golden_feats.std():.2f}")

    min_fr = min(torch_feats.shape[1], golden_feats.shape[1])
    t_f = torch_feats[:, :min_fr]
    g_f = golden_feats[:, :min_fr]
    offset = t_f.mean() - g_f.mean()
    corr = np.corrcoef(t_f.flatten(), g_f.flatten())[0, 1]

    print(f"\n  [COMPARISON over {min_fr} common frames]")
    print(f"    OFFSET (torchaudio - golden): {offset:.2f} log2 units")
    print(f"    RATIO:  {t_f.mean() / max(g_f.mean(), 0.001):.2f}x")
    print(f"    Pearson correlation: {corr:.4f}")
    print(f"    Max abs diff: {np.abs(t_f - g_f).max():.2f}")

    if abs(offset) > 5:
        print(f"\n  ** CRITICAL: Features are offset by {abs(offset):.0f} log2 units. **")
        print(f"  ** Model sees golden features as out-of-distribution noise. **")

    # ── A/B Model Inference ───────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"STEP 3: A/B Model Inference Test")
    print(f"{'─'*70}")

    pred_a, conf_a, probs_a, shape_a = infer(model, torch_feats, labels)
    print(f"\n  [A] Torchaudio features → model (MATCHES training)")
    print(f"    Tensor shape: {shape_a}")
    print(f"    Prediction: {pred_a} ({conf_a*100:.1f}%)")
    print_probs(labels, probs_a, prefix="  ")

    pred_b, conf_b, probs_b, shape_b = infer(model, golden_feats, labels)
    print(f"\n  [B] Golden features → model (USED AT INFERENCE)")
    print(f"    Tensor shape: {shape_b}")
    print(f"    Prediction: {pred_b} ({conf_b*100:.1f}%)")
    print_probs(labels, probs_b, prefix="  ")

    # Rescaled test
    scale_factor = t_f.mean() / max(g_f.mean(), 0.001)
    golden_rescaled = golden_feats * scale_factor
    pred_c, conf_c, probs_c, shape_c = infer(model, golden_rescaled, labels)
    print(f"\n  [C] Golden features x{scale_factor:.2f} → model (RESCALED to torchaudio range)")
    print(f"    Prediction: {pred_c} ({conf_c*100:.1f}%)")

    # ── Diagnosis ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  DIAGNOSIS SUMMARY")
    print(f"{'='*70}")
    if pred_a != pred_b:
        print(f"\n  ROOT CAUSE CONFIRMED: Feature extractor mismatch.")
        print(f"    - Training used LogMelExtractor (torchaudio) with int16 input.")
        print(f"      File: src/ml/Pipeline/process_data.py:280-281")
        print(f"      File: src/ml/Pipeline/features.py:56 (no /32768 normalization)")
        print(f"    - Inference uses GoldenExtractor with int14 input.")
        print(f"      File: src/ml/my_test/golden_model.py")
        print(f"    - Feature range: training [{torch_feats.min():.0f},{torch_feats.max():.0f}]"
              f" vs inference [{golden_feats.min():.0f},{golden_feats.max():.0f}]")
        print(f"    - Offset: {offset:.1f} log2 units ({scale_factor:.2f}x)")
        if pred_c == pred_a:
            print(f"\n  PROOF: Rescaling golden features by {scale_factor:.2f}x restores")
            print(f"  the correct prediction ({pred_c}, {conf_c*100:.1f}% confidence).")
    else:
        print(f"\n  Features produce same prediction. Check other causes.")

    print(f"\n  FIX: Retrain with process_data_golden.py + fix Pipeline/model.py groups.")
    print(f"  See diagnosis for details.\n")


if __name__ == "__main__":
    main()
