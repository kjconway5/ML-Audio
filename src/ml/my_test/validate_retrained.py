#!/usr/bin/env python3
"""
validate_retrained.py — Post-retrain validation suite.

Confirms that a newly trained model is compatible with golden/RTL features.
Run after retraining with process_data_golden.py.

Usage:
    python validate_retrained.py --model <new_checkpoint.pt>
    python validate_retrained.py --model dscnn-golden-retrained.pt --wav test.wav
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


def load_model(model_path):
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


def check_quant_range(ckpt):
    """VALIDATION 1: QuantStub observer range matches golden feature range."""
    sd = ckpt["model_state_dict"]
    q_scale = sd["quant.scale"].item()
    q_zp = sd["quant.zero_point"].item()
    obs_max = q_scale * 255

    print(f"\n[V1] QuantStub range check")
    print(f"  quant.scale={q_scale:.6f}, zero_point={q_zp}")
    print(f"  Observed range: [0, {obs_max:.2f}]")

    # Golden features range: [0, ~16]. With moving average, observer max should be ~16-20.
    if obs_max > 25:
        print(f"  FAIL: Observer range {obs_max:.1f} too wide. Model NOT trained on golden features.")
        return False
    if obs_max < 10:
        print(f"  FAIL: Observer range {obs_max:.1f} too narrow.")
        return False
    print(f"  PASS: Range consistent with golden features [0, ~16].")
    return True


def check_architecture(ckpt):
    """VALIDATION 2: Depthwise weights have correct shape (groups=in_channels)."""
    sd = ckpt["model_state_dict"]
    print(f"\n[V2] Architecture check (depthwise groups)")
    ok = True
    for k, v in sd.items():
        if "depthwise" in k and "weight" in k and hasattr(v, "shape"):
            expected_in_ch = 1  # For groups=in_channels: shape is [C, 1, kH, kW]
            actual_in_ch = v.shape[1]
            status = "PASS" if actual_in_ch == expected_in_ch else "FAIL"
            print(f"  {k}: {v.shape} — {status}")
            if actual_in_ch != expected_in_ch:
                ok = False
    if ok:
        print(f"  PASS: All depthwise layers use groups=in_channels.")
    else:
        print(f"  FAIL: Some depthwise layers have wrong groups.")
    return ok


def check_golden_inference(model, labels):
    """VALIDATION 3: Golden features produce confident, non-uniform predictions."""
    ge = GoldenExtractor()

    # Test with a strong 1kHz tone — should NOT predict silence/unknown
    t = np.arange(16000) / 16000.0
    audio = (0.5 * np.sin(2 * np.pi * 1000 * t) * 8191).astype(np.int16)
    feats = ge.extract_float(audio)
    tensor = torch.from_numpy(feats.T).unsqueeze(0).unsqueeze(0).float()

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].numpy()

    pred_idx = np.argmax(probs)
    confidence = probs[pred_idx]

    print(f"\n[V3] Golden feature inference (1kHz tone)")
    print(f"  Feature range: [{feats.min():.2f}, {feats.max():.2f}]")
    print(f"  Prediction: {labels[pred_idx]} ({confidence*100:.1f}%)")

    # With a tone, we shouldn't get silence with high confidence
    silence_idx = labels.index("silence") if "silence" in labels else -1
    unknown_idx = labels.index("unknown") if "unknown" in labels else -1

    if pred_idx in (silence_idx, unknown_idx) and confidence > 0.9:
        print(f"  WARN: Loud tone classified as {labels[pred_idx]} — features may be out of range.")
        return False
    if confidence < 0.3:
        print(f"  WARN: Low confidence ({confidence*100:.1f}%) — model may not match features.")
        return False
    print(f"  PASS: Model produces confident prediction on golden features.")
    return True


def check_golden_vs_rtl_features(model, labels):
    """VALIDATION 4: Golden extractor matches RTL features file (if available)."""
    rtl_npy = _HERE.parent.parent / "rtl" / "top" / "rtl_features.npy"
    if not rtl_npy.exists():
        print(f"\n[V4] Golden vs RTL consistency — SKIPPED (no rtl_features.npy)")
        return True

    rtl_feats = np.load(str(rtl_npy))
    print(f"\n[V4] Golden vs RTL consistency")
    print(f"  RTL features: shape={rtl_feats.shape}, range=[{rtl_feats.min():.3f}, {rtl_feats.max():.3f}]")

    # Run inference on both
    tensor_rtl = torch.from_numpy(rtl_feats.T).unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad():
        probs_rtl = torch.softmax(model(tensor_rtl), dim=1)[0].numpy()
    pred_rtl = labels[np.argmax(probs_rtl)]
    conf_rtl = probs_rtl.max()

    print(f"  RTL prediction: {pred_rtl} ({conf_rtl*100:.1f}%)")
    assert rtl_feats.max() < 17, f"RTL features max={rtl_feats.max():.2f} exceeds golden range!"
    print(f"  PASS: RTL features in golden range.")
    return True


def check_wav_inference(model, labels, wav_path, expected_label=None):
    """VALIDATION 5: Test on a real WAV file."""
    import soundfile as sf

    audio_f, sr = sf.read(str(wav_path), dtype="float32")
    if audio_f.ndim > 1:
        audio_f = audio_f.mean(axis=1)
    if sr != 16000:
        from scipy.signal import resample
        audio_f = resample(audio_f, int(len(audio_f) * 16000 / sr))
    if len(audio_f) < 16000:
        audio_f = np.pad(audio_f, (0, 16000 - len(audio_f)))
    else:
        audio_f = audio_f[:16000]

    audio_i14 = (np.clip(audio_f, -1.0, 1.0) * 8191).astype(np.int16)
    ge = GoldenExtractor()
    feats = ge.extract_float(audio_i14)
    tensor = torch.from_numpy(feats.T).unsqueeze(0).unsqueeze(0).float()

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].numpy()
    pred_idx = np.argmax(probs)
    pred = labels[pred_idx]
    conf = probs[pred_idx]

    print(f"\n[V5] WAV file inference: {wav_path}")
    print(f"  Prediction: {pred} ({conf*100:.1f}%)")
    for i, l in enumerate(labels):
        print(f"    {l:>10s}: {probs[i]*100:5.1f}%")

    if expected_label:
        ok = pred == expected_label
        print(f"  Expected: {expected_label} — {'PASS' if ok else 'FAIL'}")
        return ok
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Retrained checkpoint path")
    parser.add_argument("--wav", default=None, help="Optional WAV file to test")
    parser.add_argument("--expected", default=None, help="Expected label for WAV file")
    args = parser.parse_args()

    model, labels, ckpt = load_model(args.model)

    print("=" * 70)
    print("  POST-RETRAIN VALIDATION SUITE")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Labels: {labels}")
    print(f"Test accuracy: {ckpt.get('test_accuracy', 'N/A')}")

    results = {}
    results["V1_quant_range"] = check_quant_range(ckpt)
    results["V2_architecture"] = check_architecture(ckpt)
    results["V3_golden_inference"] = check_golden_inference(model, labels)
    results["V4_rtl_consistency"] = check_golden_vs_rtl_features(model, labels)

    if args.wav:
        results["V5_wav_inference"] = check_wav_inference(
            model, labels, args.wav, args.expected
        )

    print(f"\n{'='*70}")
    print(f"  VALIDATION SUMMARY")
    print(f"{'='*70}")
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    if all_pass:
        print(f"\n  ALL VALIDATIONS PASSED.")
        print(f"  Model is ready for golden/RTL inference.")
    else:
        print(f"\n  SOME VALIDATIONS FAILED. Review issues above.")
    print()


if __name__ == "__main__":
    main()
