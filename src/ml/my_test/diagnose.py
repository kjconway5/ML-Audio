#!/usr/bin/env python3
"""
Diagnosis script: identifies exact training vs inference feature mismatch.

Run from project root:
    python src/ml/my_test/diagnose.py
    python src/ml/my_test/diagnose.py --wav path/to/test.wav
"""
import sys
import numpy as np
import torch
from pathlib import Path

# Add paths
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "Pipeline"))

from golden_model import GoldenExtractor
from features import LogMelExtractor


def compare_extractors(audio_int16: np.ndarray, label: str = "test"):
    """Compare training (torchaudio) vs inference (golden) features."""
    print(f"\n{'='*70}")
    print(f"  DIAGNOSIS: {label}")
    print(f"{'='*70}")

    # --- Input stats ---
    print(f"\n[INPUT AUDIO]")
    print(f"  Shape: {audio_int16.shape}, dtype: {audio_int16.dtype}")
    print(f"  Range: [{audio_int16.min()}, {audio_int16.max()}]")
    rms = np.sqrt(np.mean(audio_int16.astype(np.float64)**2))
    print(f"  RMS: {rms:.1f}")
    n_clipped = np.sum(np.abs(audio_int16) > 8191)
    print(f"  Samples |val|>8191 (14-bit clip): {n_clipped}/{len(audio_int16)} "
          f"({100*n_clipped/len(audio_int16):.1f}%)")

    # --- Training extractor (LogMelExtractor / torchaudio) ---
    train_ext = LogMelExtractor(
        sample_rate=16000, n_fft=256, n_mels=40,
        hop_length=128, window_length=256
    )
    train_feats, _ = train_ext.extract(audio_int16)

    print(f"\n[TRAINING EXTRACTOR — LogMelExtractor (torchaudio)]")
    print(f"  Shape: {train_feats.shape}  (n_mels={train_feats.shape[0]}, "
          f"n_frames={train_feats.shape[1]})")
    print(f"  dtype: {train_feats.dtype}")
    print(f"  Range: [{train_feats.min():.4f}, {train_feats.max():.4f}]")
    print(f"  Mean:  {train_feats.mean():.4f}")
    print(f"  Std:   {train_feats.std():.4f}")

    # --- Inference extractor (GoldenExtractor) ---
    try:
        golden_ext = GoldenExtractor()
    except AssertionError as e:
        print(f"\n[INFERENCE EXTRACTOR — GoldenExtractor]")
        print(f"  ** CANNOT INSTANTIATE: {e} **")
        print(f"  Fix: regenerate hanning.hex as 256-point periodic Hann")
        print(f"  Run: python -c \"import math; "
              f"open('src/rtl/STFFT/hanning.hex','w').writelines("
              f"f'{{int(0.5*(1-math.cos(2*3.14159265*n/256))*2047+0.5):03x}}\\n' "
              f"for n in range(256))\"")
        return None, None

    golden_feats = golden_ext.extract_float(audio_int16)

    print(f"\n[INFERENCE EXTRACTOR — GoldenExtractor (RTL bit-accurate)]")
    print(f"  Shape: {golden_feats.shape}  (n_mels={golden_feats.shape[0]}, "
          f"n_frames={golden_feats.shape[1]})")
    print(f"  dtype: {golden_feats.dtype}")
    print(f"  Range: [{golden_feats.min():.4f}, {golden_feats.max():.4f}]")
    print(f"  Mean:  {golden_feats.mean():.4f}")
    print(f"  Std:   {golden_feats.std():.4f}")

    # --- Compare ---
    min_frames = min(train_feats.shape[1], golden_feats.shape[1])
    t = train_feats[:, :min_frames]
    g = golden_feats[:, :min_frames]
    diff = np.abs(t - g)

    print(f"\n[FEATURE COMPARISON — aligned to {min_frames} common frames]")
    print(f"  Frame count: training={train_feats.shape[1]}, golden={golden_feats.shape[1]}")
    print(f"  Max abs diff:    {diff.max():.4f}")
    print(f"  Mean abs diff:   {diff.mean():.4f}")
    print(f"  Median abs diff: {np.median(diff):.4f}")
    print(f"  >1.0 diff: {(diff>1).sum()}/{diff.size} ({100*(diff>1).sum()/diff.size:.1f}%)")
    print(f"  >5.0 diff: {(diff>5).sum()}/{diff.size} ({100*(diff>5).sum()/diff.size:.1f}%)")
    print(f"  >10.0 diff: {(diff>10).sum()}/{diff.size} ({100*(diff>10).sum()/diff.size:.1f}%)")

    corr = np.corrcoef(t.flatten(), g.flatten())[0, 1]
    print(f"  Pearson correlation: {corr:.6f}")

    scale_offset = t.mean() - g.mean()
    print(f"\n  ** SCALE OFFSET (training - golden): {scale_offset:.2f} log2 units **")
    if abs(scale_offset) > 2.0:
        print(f"  !! CRITICAL: Features are offset by ~{abs(scale_offset):.0f} log2 units.")
        print(f"     Model trained on ~{t.mean():.0f} sees ~{g.mean():.0f} at inference.")
        print(f"     This alone explains poor accuracy.")

    # --- Tensor shape check ---
    print(f"\n[TENSOR SHAPES]")
    # Training path: feats.T -> (T,M), unsqueeze(0) -> (1,T,M), batched -> (B,1,T,M)
    train_tensor = torch.from_numpy(train_feats.T).unsqueeze(0).unsqueeze(0).float()
    print(f"  Training (batched): {train_tensor.shape}  (B,C,T,M)")

    # Inference path: features.T -> (T,M), unsqueeze(0).unsqueeze(0) -> (1,1,T,M)
    infer_tensor = torch.from_numpy(golden_feats.T).unsqueeze(0).unsqueeze(0).float()
    print(f"  Inference:          {infer_tensor.shape}  (B,C,T,M)")
    if train_tensor.shape[2:] != infer_tensor.shape[2:]:
        print(f"  !! SHAPE MISMATCH: T differs ({train_tensor.shape[2]} vs {infer_tensor.shape[2]})")
    else:
        print(f"  Shape convention: MATCH")

    return train_feats, golden_feats


def test_model_with_both_features(model_path: str, audio_int16: np.ndarray):
    """Run the model with both training and golden features."""
    from dscnn import DSCNN

    print(f"\n{'='*70}")
    print(f"  MODEL INFERENCE COMPARISON")
    print(f"{'='*70}")

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    labels = checkpoint.get('labels', ['go','left','no','right','silence','unknown','yes'])
    is_quantized = checkpoint.get('quantized', False)

    model_cfg = config.get('model', {})
    preproc_cfg = config.get('preprocessing', {})
    model = DSCNN(
        n_classes=len(labels),
        n_mels=preproc_cfg.get('n_mels', 40),
        first_conv_filters=model_cfg.get('first_conv', {}).get('filters', 24),
        first_conv_kernel=tuple(model_cfg.get('first_conv', {}).get('kernel_size', [10, 4])),
        first_conv_stride=tuple(model_cfg.get('first_conv', {}).get('stride', [2, 2])),
        n_ds_blocks=model_cfg.get('ds_blocks', {}).get('n_blocks', 4),
        ds_filters=model_cfg.get('ds_blocks', {}).get('filters', 24),
        ds_kernel=tuple(model_cfg.get('ds_blocks', {}).get('kernel_size', [3, 3])),
        ds_stride=tuple(model_cfg.get('ds_blocks', {}).get('stride', [1, 1])),
    )

    if is_quantized:
        qat_backend = checkpoint.get('qat_backend', 'fbgemm')
        model.eval()
        model.fuse_model()
        model.train()
        model.qconfig = torch.quantization.get_default_qat_qconfig(qat_backend)
        torch.quantization.prepare_qat(model, inplace=True)
        model.eval()
        torch.quantization.convert(model, inplace=True)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Training features
    train_ext = LogMelExtractor(
        sample_rate=16000, n_fft=256, n_mels=40,
        hop_length=128, window_length=256
    )
    train_feats, _ = train_ext.extract(audio_int16)
    train_tensor = torch.from_numpy(train_feats.T).unsqueeze(0).unsqueeze(0).float()

    with torch.no_grad():
        logits_train = model(train_tensor)
        probs_train = torch.softmax(logits_train, dim=1)[0].numpy()

    pred_train = np.argmax(probs_train)
    print(f"\n[MODEL + TRAINING FEATURES]")
    print(f"  Input shape: {train_tensor.shape}")
    print(f"  Prediction: {labels[pred_train]} ({probs_train[pred_train]*100:.1f}%)")
    for i, l in enumerate(labels):
        print(f"    {l:>10s}: {probs_train[i]*100:5.1f}%")

    # Golden features
    try:
        golden_ext = GoldenExtractor()
        golden_feats = golden_ext.extract_float(audio_int16)
        golden_tensor = torch.from_numpy(golden_feats.T).unsqueeze(0).unsqueeze(0).float()

        with torch.no_grad():
            logits_golden = model(golden_tensor)
            probs_golden = torch.softmax(logits_golden, dim=1)[0].numpy()

        pred_golden = np.argmax(probs_golden)
        print(f"\n[MODEL + GOLDEN FEATURES]")
        print(f"  Input shape: {golden_tensor.shape}")
        print(f"  Prediction: {labels[pred_golden]} ({probs_golden[pred_golden]*100:.1f}%)")
        for i, l in enumerate(labels):
            print(f"    {l:>10s}: {probs_golden[i]*100:5.1f}%")

        print(f"\n[VERDICT]")
        if pred_train == pred_golden:
            print(f"  Same prediction: {labels[pred_train]}")
        else:
            print(f"  !! DIFFERENT PREDICTIONS:")
            print(f"     Training features → {labels[pred_train]}")
            print(f"     Golden features   → {labels[pred_golden]}")
            print(f"     ROOT CAUSE CONFIRMED: feature extractor mismatch")
    except AssertionError as e:
        print(f"\n[MODEL + GOLDEN FEATURES]")
        print(f"  SKIPPED — GoldenExtractor failed: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Diagnose training vs inference mismatch")
    parser.add_argument('--wav', type=str, help='Path to WAV file')
    parser.add_argument('--model', type=str,
                        default=str(_HERE / 'dscnn-golden.pt'),
                        help='Model checkpoint')
    args = parser.parse_args()

    if args.wav:
        import soundfile as sf
        audio, sr = sf.read(args.wav, dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            from scipy import signal
            audio = signal.resample(audio, int(len(audio) * 16000 / sr))
        if len(audio) < 16000:
            audio = np.pad(audio, (0, 16000 - len(audio)))
        else:
            audio = audio[:16000]
        audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        label = Path(args.wav).stem
    else:
        # Generate synthetic test audio
        np.random.seed(42)
        t = np.arange(16000) / 16000.0
        audio = 0.3 * np.sin(2 * np.pi * 300 * t) + 0.1 * np.random.randn(16000)
        audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        label = "synthetic_300Hz"

    compare_extractors(audio_int16, label)
    test_model_with_both_features(args.model, audio_int16)


if __name__ == "__main__":
    main()
