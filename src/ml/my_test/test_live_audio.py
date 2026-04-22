#!/usr/bin/env python3
"""
Keyword Spotting Test

Records 1 second of audio from the microphone and classifies it.

Usage:
    python test_live_audio.py                           # Use default model
    python test_live_audio.py -m model_final.pt         # Specify model
    python test_live_audio.py -m model.pt -f audio.wav  # Test WAV file
    python test_live_audio.py --list-devices            # List audio devices
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dscnn import DSCNN
from golden_model import GoldenExtractor

# 14-bit ADC range — must match process_data.py float_to_int14()
_SAMPLE_MAX = (1 << 13) - 1  # 8191

# Default paths
DEFAULT_MODEL_PATH = Path(__file__).parent / "model_final.pt"


def load_model(model_path, device='cpu'):
    """Load trained model and return classifier components."""
    import json

    print(f"\n{'='*50}")
    print(f"Loading model: {Path(model_path).name}")
    print(f"{'='*50}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Get config and labels
    config = checkpoint['config']
    is_quantized = checkpoint.get('quantized', False)

    # Load labels from checkpoint (saved during training)
    if 'labels' in checkpoint:
        labels = checkpoint['labels']
        print(f"  Labels from: checkpoint")
    else:
        # Fallback for older checkpoints: use alphabetically sorted
        labels = sorted(config.get('data', {}).get('classes', ['silence', 'unknown', 'yes']))
        print(f"  Labels (sorted fallback): {labels}")

    # Get pipeline config
    pipeline_cfg = checkpoint.get('pipeline') or config.get('pipeline', {})
    feature_cfg = pipeline_cfg.get('feature_extractor', {})
    preproc_cfg = config.get('preprocessing', {})
    if not feature_cfg:
        feature_cfg = preproc_cfg

    # Build model
    model_cfg = config.get('model', {})
    model = DSCNN(
        n_classes=len(labels),
        n_mels=feature_cfg.get('n_mels', 40),
        first_conv_filters=model_cfg.get('first_conv', {}).get('filters', 32),
        first_conv_kernel=tuple(model_cfg.get('first_conv', {}).get('kernel_size', [10, 4])),
        first_conv_stride=tuple(model_cfg.get('first_conv', {}).get('stride', [2, 2])),
        n_ds_blocks=model_cfg.get('ds_blocks', {}).get('n_blocks', 4),
        ds_filters=model_cfg.get('ds_blocks', {}).get('filters', 32),
        ds_kernel=tuple(model_cfg.get('ds_blocks', {}).get('kernel_size', [3, 3])),
        ds_stride=tuple(model_cfg.get('ds_blocks', {}).get('stride', [1, 1])),
    )

    # Handle quantized models - must use QAT config to match training
    if is_quantized:
        qat_backend = checkpoint.get('qat_backend', 'fbgemm')
        # Fuse layers in eval mode
        model.eval()
        model.fuse_model()
        # prepare_qat requires train mode
        model.train()
        model.qconfig = torch.quantization.get_default_qat_qconfig(qat_backend)
        torch.quantization.prepare_qat(model, inplace=True)
        # Convert to quantized model
        model.eval()
        torch.quantization.convert(model, inplace=True)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create feature extractor — must match training pipeline
    pipeline = GoldenExtractor()

    # Print info
    print(f"  Model type: {'INT8 Quantized' if is_quantized else 'Float32'}")
    print(f"  Classes: {labels}")
    test_acc = checkpoint.get('test_accuracy')
    if test_acc is not None:
        print(f"  Test accuracy: {test_acc:.2f}%")
    print(f"{'='*50}")

    return model, pipeline, labels, device


def _to_int14(audio_float: np.ndarray) -> np.ndarray:
    """Convert float32 [-1, 1] audio to 14-bit signed range — matches process_data.py."""
    return (np.clip(audio_float, -1.0, 1.0) * _SAMPLE_MAX).astype(np.int16)


def _compute_energy(audio_int14: np.ndarray) -> float:
    """Mean squared energy in normalized float space — used for VAD threshold."""
    x = audio_int14.astype(np.float32) / _SAMPLE_MAX
    return float(np.mean(x ** 2))


def calibrate_noise_floor(duration=2.0, sample_rate=16000):
    """Record ambient noise and return mean squared energy for VAD threshold."""
    print(f"\n{'='*50}")
    print(f"  VAD CALIBRATION")
    print(f"  Stay quiet for {duration:.1f}s while we measure ambient noise...")
    print(f"{'='*50}")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype=np.float32,
    )
    sd.wait()
    audio_int14 = _to_int14(audio[:, 0])
    noise_energy = _compute_energy(audio_int14)
    noise_rms = np.sqrt(noise_energy)
    print(f"  Noise floor RMS: {noise_rms:.6f} (energy: {noise_energy:.8f})")
    print(f"{'='*50}")
    return noise_energy


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


def preprocess_audio(audio, extractor, energy_threshold=None):
    """Preprocess audio through GoldenExtractor.

    Returns None if energy is below threshold (VAD gate).
    """
    audio_int14 = _to_int14(audio)

    # VAD energy gate
    if energy_threshold is not None:
        if _compute_energy(audio_int14) < energy_threshold:
            return None

    # Extract features: (n_mels, n_frames) -> (1, 1, n_frames, n_mels)
    features = extractor.extract_float(audio_int14)
    features_tensor = torch.from_numpy(features.T).unsqueeze(0).unsqueeze(0).float()
    return features_tensor


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


def test_wav_file(model, extractor, labels, file_path, energy_threshold=None):
    """Test on a WAV file."""
    import soundfile as sf

    print(f"\nTesting file: {file_path}")
    audio, sr = sf.read(str(file_path), dtype='float32')

    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    if sr != 16000:
        from scipy import signal
        audio = signal.resample(audio, int(len(audio) * 16000 / sr))

    target_length = 16000
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]

    features = preprocess_audio(audio, extractor, energy_threshold)

    if features is None:
        pred_idx = labels.index("silence") if "silence" in labels else 0
        probs = np.zeros(len(labels))
        probs[pred_idx] = 1.0
        print_results(pred_idx, 1.0, probs, labels)
        print(f"  (energy gate — below threshold, skipped model)\n")
        return pred_idx, 1.0

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
    parser.add_argument(
        '--vad-multiplier',
        type=float,
        default=3.0,
        help='VAD threshold = noise_floor_energy * multiplier^2 (default: 3.0)'
    )
    parser.add_argument(
        '--vad-duration',
        type=float,
        default=2.0,
        help='Duration in seconds for VAD calibration (default: 2.0)'
    )
    parser.add_argument(
        '--vad-threshold',
        type=float,
        default=0.0001,
        help='Default energy threshold for file mode (~-40dB, default: 0.0001). '
             'In mic mode this is overridden by calibration.'
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

    # --- VAD setup ---
    if args.file:
        energy_threshold = args.vad_threshold
        print(f"\n  VAD energy threshold: {energy_threshold:.6f} (default)")
    else:
        noise_energy = calibrate_noise_floor(
            duration=args.vad_duration, sample_rate=16000
        )
        energy_threshold = noise_energy * (args.vad_multiplier ** 2)
        print(f"  VAD energy threshold: {energy_threshold:.8f} "
              f"(noise {noise_energy:.8f} x {args.vad_multiplier}^2)\n")

    # --- Get audio ---
    if args.file:
        if not Path(args.file).exists():
            print(f"Error: File not found: {args.file}")
            sys.exit(1)
        test_wav_file(model, extractor, labels, args.file, energy_threshold)
    else:
        audio = record_audio(duration=1.0, sample_rate=16000)

        start_time = time.time()
        features = preprocess_audio(audio, extractor, energy_threshold)
        inference_time = (time.time() - start_time) * 1000

        if features is None:
            pred_idx = labels.index("silence") if "silence" in labels else 0
            probs = np.zeros(len(labels))
            probs[pred_idx] = 1.0
            print_results(pred_idx, 1.0, probs, labels)
            print(f"  (energy gate — below threshold, skipped model)\n")
        else:
            pred_idx, confidence, probs = classify(model, features, labels)
            print_results(pred_idx, confidence, probs, labels)
            print(f"  Inference time: {inference_time:.1f}ms\n")


if __name__ == "__main__":
    main()