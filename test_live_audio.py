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
from model import DSCNN
from pipeline import SimplePipeline

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

    # Create pipeline
    pipeline = SimplePipeline(
        sample_rate=feature_cfg.get('sample_rate', 16000),
        use_filters=pipeline_cfg.get('use_filters', False),
        n_mels=feature_cfg.get('n_mels', 40),
        n_fft=feature_cfg.get('n_fft', 256),
        hop_length=feature_cfg.get('hop_length', 160),
        window_length=feature_cfg.get('window_length', 256),
    )

    # Print info
    print(f"  Model type: {'INT8 Quantized' if is_quantized else 'Float32'}")
    print(f"  Classes: {labels}")
    test_acc = checkpoint.get('test_accuracy')
    if test_acc is not None:
        print(f"  Test accuracy: {test_acc:.2f}%")
    print(f"{'='*50}")

    return model, pipeline, labels, device


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


def preprocess_audio(audio, pipeline):
    """Preprocess audio through the pipeline."""
    # Convert to int16 (expected by pipeline)
    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)

    # Run through pipeline
    features, _ = pipeline.process(audio_int16)

    # Features shape: (n_mels, time) -> (batch, channel, time, mels)
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


def test_wav_file(model, pipeline, labels, file_path):
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

    # Pad or trim to 1 second
    target_length = 16000
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]

    # Classify
    features = preprocess_audio(audio, pipeline)
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
    model, pipeline, labels, device = load_model(args.model)

    # Test WAV file or record from mic
    if args.file:
        if not Path(args.file).exists():
            print(f"Error: File not found: {args.file}")
            sys.exit(1)
        test_wav_file(model, pipeline, labels, args.file)
    else:
        # Record from microphone
        audio = record_audio(duration=1.0, sample_rate=16000)

        # Classify
        start_time = time.time()
        features = preprocess_audio(audio, pipeline)
        pred_idx, confidence, probs = classify(model, features, labels)
        inference_time = (time.time() - start_time) * 1000

        # Print results
        print_results(pred_idx, confidence, probs, labels)
        print(f"  Inference time: {inference_time:.1f}ms\n")


if __name__ == "__main__":
    main()
