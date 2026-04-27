#!/usr/bin/env python3
"""
Keyword Spotting Test

Records 1 second of audio from the microphone and classifies it.
Uses GoldenExtractor (RTL-accurate features) to match training pipeline.

Usage:
    python test_live_audio.py                                      # Use default model
    python test_live_audio.py -m tiny-7class-golden.pt             # Specify model
    python test_live_audio.py -m tiny-7class-golden.pt -f audio.wav  # Test WAV file
    python test_live_audio.py --list-devices                       # List audio devices
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from model import DSCNN
from golden_model import GoldenExtractor

# Default paths
DEFAULT_MODEL_PATH = Path(__file__).parent / "tiny-7class-golden.pt"
SAMPLE_MAX = (1 << 13) - 1  # 8191 (14-bit signed, matches RTL ADC)


def load_model(model_path, device='cpu'):
    """Load trained model and return classifier components."""
    print(f"\n{'='*50}")
    print(f"Loading model: {Path(model_path).name}")
    print(f"{'='*50}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Get config and labels
    config = checkpoint['config']
    is_quantized = checkpoint.get('quantized', False)

    # Load labels from checkpoint (saved during training)
    if 'labels' in checkpoint:
        labels = checkpoint['labels']
        print(f"  Labels from: checkpoint")
    else:
        labels = sorted(config.get('data', {}).get('classes', []))
        print(f"  Labels (sorted fallback): {labels}")

    preproc_cfg = config.get('preprocessing', {})
    model_cfg = config.get('model', {})

    # Build model
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

    # Handle quantized models - must use QAT config to match training
    if is_quantized:
        qat_backend = checkpoint.get('qat_backend', 'fbgemm')
        model.eval()
        model.fuse_model()
        model.train()
        model.qconfig = torch.quantization.get_default_qat_qconfig(qat_backend)
        torch.quantization.prepare_qat(model, inplace=True)
        model.eval()
        torch.quantization.convert(model, inplace=True)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create GoldenExtractor (RTL-accurate feature pipeline)
    extractor = GoldenExtractor()

    # Print info
    print(f"  Model type: {'INT8 Quantized' if is_quantized else 'Float32'}")
    print(f"  Classes: {labels}")
    print(f"  Features: GoldenExtractor (RTL-accurate)")
    test_acc = checkpoint.get('test_accuracy')
    if test_acc is not None:
        print(f"  Test accuracy: {test_acc:.2f}%")
    print(f"{'='*50}")

    return model, extractor, labels, device


def record_audio(duration=1.0, sample_rate=16000):
    """Record audio from microphone."""
    print("\n🎤 Speak now...", end=" ", flush=True)
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype=np.float32,
    )
    sd.wait()
    print("Done!")
    return audio[:, 0]


def preprocess_audio(audio, extractor):
    """Preprocess audio through the GoldenExtractor (RTL-accurate pipeline)."""
    # Pad or trim to 1 second
    N = 16000
    if len(audio) < N:
        audio = np.pad(audio, (0, N - len(audio)))
    else:
        audio = audio[:N]

    # Convert to 14-bit signed integer (matches RTL ADC input)
    audio_i14 = (np.clip(audio, -1.0, 1.0) * SAMPLE_MAX).astype(np.int16)

    # Extract features using GoldenExtractor
    feats = extractor.extract_float(audio_i14)  # (n_mels, n_frames)

    # Transpose to (n_frames, n_mels)
    x = torch.from_numpy(feats.T).float()

    # Per-utterance mean/std normalization (must match training)
    x = x - x.mean()
    std = x.std()
    if std > 1e-6:
        x = x / std

    # (1, 1, n_frames, n_mels)
    return x.unsqueeze(0).unsqueeze(0)


def classify(model, features, labels):
    """Run inference and return prediction."""
    with torch.no_grad():
        logits = model(features)
        probs = torch.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()

    return pred_idx, confidence, probs[0].numpy()


def print_results(pred_idx, confidence, probs, labels):
    """Print classification results."""
    pred_label = labels[pred_idx]

    print(f"\n{'='*50}")
    print(f"  PREDICTION: {pred_label.upper()}")
    print(f"  CONFIDENCE: {confidence*100:.1f}%")
    print(f"{'='*50}")

    print("\n  Class Probabilities:")
    sorted_indices = np.argsort(probs)[::-1]
    for idx in sorted_indices:
        label = labels[idx]
        prob = probs[idx]
        bar_len = int(prob * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        marker = " ◄" if idx == pred_idx else ""
        print(f"    {label:>10s}: {prob*100:5.1f}% |{bar}|{marker}")
    print()


def test_wav_file(model, extractor, labels, file_path):
    """Test on a WAV file."""
    import soundfile as sf

    print(f"\nTesting file: {file_path}")
    audio, sr = sf.read(str(file_path), dtype='float32')

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != 16000:
        from scipy import signal
        audio = signal.resample(audio, int(len(audio) * 16000 / sr))

    # Classify (pad/trim handled in preprocess_audio)
    features = preprocess_audio(audio, extractor)
    pred_idx, confidence, probs = classify(model, features, labels)
    print_results(pred_idx, confidence, probs, labels)

    return pred_idx, confidence


def list_audio_devices():
    """Print available audio devices."""
    print("\nAvailable audio devices:")
    print("-" * 60)
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        marker = ">>>" if device["max_input_channels"] > 0 else "   "
        print(f"{marker} [{i}] {device['name']}")
    print("-" * 60)
    print("Devices with >>> have microphone input\n")


def main():
    parser = argparse.ArgumentParser(
        description="Test keyword spotting model with microphone or WAV file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_live_audio.py -m model_final.pt           # Record and classify
  python test_live_audio.py -m model_final.pt -f test.wav  # Test WAV file
  python test_live_audio.py --list-devices              # List audio devices
        """
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help='Path to .pt model file (default: model_final.pt)'
    )
    parser.add_argument(
        '--file', '-f',
        type=str,
        default=None,
        help='Path to WAV file to classify (optional)'
    )
    parser.add_argument(
        '--list-devices',
        action='store_true',
        help='List available audio devices and exit'
    )
    parser.add_argument(
        '--device',
        type=int,
        default=None,
        help='Audio device index to use'
    )
    args = parser.parse_args()

    # List devices if requested
    if args.list_devices:
        list_audio_devices()
        return

    # Set audio device if specified
    if args.device is not None:
        sd.default.device = args.device

    # Check if model exists
    if not Path(args.model).exists():
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)

    # Load model
    model, extractor, labels, device = load_model(args.model)

    # Test WAV file or record from mic
    if args.file:
        if not Path(args.file).exists():
            print(f"Error: File not found: {args.file}")
            sys.exit(1)
        test_wav_file(model, extractor, labels, args.file)
    else:
        # Record from microphone
        audio = record_audio(duration=1.0, sample_rate=16000)

        # Classify
        start_time = time.time()
        features = preprocess_audio(audio, extractor)
        pred_idx, confidence, probs = classify(model, features, labels)
        inference_time = (time.time() - start_time) * 1000

        # Print results
        print_results(pred_idx, confidence, probs, labels)
        print(f"  Inference time: {inference_time:.1f}ms\n")


if __name__ == "__main__":
    main()
