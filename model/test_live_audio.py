#!/usr/bin/env python3
"""
Live Audio Testing for Speech Commands Model

Test your trained model with:
  - Live microphone input (press Enter to record)
  - Pre-recorded WAV files

Usage:
  python test_live_audio.py                      # Live mic (interactive)
  python test_live_audio.py --file audio.wav     # Test WAV file
  python test_live_audio.py --folder test_wavs/  # Test folder of WAVs
  python test_live_audio.py --list-devices       # List audio devices
"""

import sys
import time
import torch
import numpy as np
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from model import DSCNN
from pipeline import SimplePipeline

# Default paths
DEFAULT_MODEL_PATH = Path(__file__).parent / "model_final.pt"


class AudioClassifier:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, device='cpu'):
        """Initialize the audio classifier."""
        print(f"Loading model from {model_path}...")

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        # Get config and labels
        self.config = checkpoint['config']

        # Priority: checkpoint top-level labels > config.labels > config.data.classes
        # checkpoint['labels'] comes from process_data.py's sorted label list (correct order)
        self.labels = (
            checkpoint.get('labels') or
            self.config.get('labels') or
            self.config.get('data', {}).get('classes') or
            ['no', 'silence', 'unknown']
        )
        self.label_to_id = (
            checkpoint.get('label_to_id') or
            self.config.get('label_to_id') or
            {l: i for i, l in enumerate(self.labels)}
        )

        # Get pipeline config - prefer checkpoint-level pipeline settings (actual training params)
        pipeline_cfg = (
            checkpoint.get('pipeline') or
            self.config.get('pipeline', {})
        )
        feature_cfg = pipeline_cfg.get('feature_extractor', {})
        filter_cfg = pipeline_cfg.get('filter', {})

        # Also check preprocessing config as fallback
        preproc_cfg = self.config.get('preprocessing', {})
        if not feature_cfg:
            feature_cfg = preproc_cfg

        # Build model from config
        model_cfg = self.config.get('model', {})
        self.model = DSCNN(
            n_classes=len(self.labels),
            n_mels=feature_cfg.get('n_mels', 64),
            first_conv_filters=model_cfg.get('first_conv', {}).get('filters', 64),
            first_conv_kernel=tuple(model_cfg.get('first_conv', {}).get('kernel_size', [10, 4])),
            first_conv_stride=tuple(model_cfg.get('first_conv', {}).get('stride', [2, 2])),
            n_ds_blocks=model_cfg.get('ds_blocks', {}).get('n_blocks', 4),
            ds_filters=model_cfg.get('ds_blocks', {}).get('filters', 64),
            ds_kernel=tuple(model_cfg.get('ds_blocks', {}).get('kernel_size', [3, 3])),
            ds_stride=tuple(model_cfg.get('ds_blocks', {}).get('stride', [1, 1])),
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.device = torch.device(device)
        self.model = self.model.to(self.device)

        # Initialize preprocessor using the SAME settings as training
        # feature_cfg comes from the pipeline that process_data.py actually used
        self.pipeline = SimplePipeline(
            sample_rate=feature_cfg.get('sample_rate', 16000),
            use_filters=pipeline_cfg.get('use_filters', False),
            n_mels=feature_cfg.get('n_mels', 40),
            n_fft=feature_cfg.get('n_fft', 512),
            hop_length=feature_cfg.get('hop_length', 160),
            window_length=feature_cfg.get('window_length',
                                          feature_cfg.get('n_fft', 512)),
            hpf_order=filter_cfg.get('hpf_order', 2),
            lpf_order=filter_cfg.get('lpf_order', 4),
            cutoff_hpf=filter_cfg.get('cutoff_hpf', 150),
            cutoff_lpf=filter_cfg.get('cutoff_lpf', 4000),
        )

        self.sample_rate = 16000
        self.duration = 1.0  # 1 second

        print(f"Model loaded on {self.device}")
        print(f"Classes: {self.labels}")
        print(f"Label mapping: {self.label_to_id}")
        print(f"Pipeline config: n_fft={feature_cfg.get('n_fft')}, "
              f"window_length={feature_cfg.get('window_length')}, "
              f"n_mels={feature_cfg.get('n_mels')}")
        print(f"Test accuracy: {checkpoint.get('test_accuracy', 'N/A'):.2%}" if isinstance(checkpoint.get('test_accuracy'), float) else "")

    def preprocess_audio(self, audio):
        """Preprocess audio through the pipeline."""
        # Ensure float32 and normalize
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Normalize to [-1, 1] if not already
        if np.abs(audio).max() > 1.0:
            audio = audio / 32768.0

        # Convert to int16 (expected by pipeline)
        audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)

        # Run through preprocessing pipeline
        features, _ = self.pipeline.process(audio_int16)

        # Features shape: (n_mels, time) -> (batch, channel, time, mels)
        # Model expects: (batch, 1, time, n_mels)
        features_tensor = torch.from_numpy(features.T).unsqueeze(0).unsqueeze(0).float()
        return features_tensor.to(self.device)

    def classify(self, audio):
        """Classify audio and return prediction."""
        features = self.preprocess_audio(audio)

        with torch.no_grad():
            outputs = self.model(features)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        predicted_idx = predicted.item()
        confidence_val = confidence.item()
        predicted_label = self.labels[predicted_idx]

        return predicted_label, confidence_val, probabilities[0].cpu().numpy()

    def print_results(self, predicted_label, confidence, probabilities):
        """Print classification results."""
        print(f"\n  ✓ Prediction: {predicted_label.upper()}")
        print(f"    Confidence: {confidence*100:.1f}%")
        print("\n  All classes:")
        for i, label in enumerate(self.labels):
            bar = "█" * int(probabilities[i] * 20)
            print(f"    {label:>10s}: {probabilities[i]*100:5.1f}% {bar}")

    def record_audio(self):
        """Record 1 second of audio from microphone."""
        import sounddevice as sd

        print("  Recording... (speak now!)")
        audio = sd.rec(
            int(self.duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        return audio.flatten()

    def run_interactive(self):
        """Run interactive mode - record and classify continuously."""
        import sounddevice as sd

        print("\n" + "=" * 60)
        print("Live Audio Classifier")
        print("=" * 60)
        try:
            print(f"Device: {sd.query_devices(sd.default.device[0])['name']}")
        except:
            print("Device: Default microphone")
        print(f"Sample rate: {self.sample_rate} Hz")
        print(f"Recording duration: {self.duration}s")
        print(f"\nRecognizable classes: {', '.join(self.labels)}")
        print("\nPress Enter to record, Ctrl+C to exit")
        print("=" * 60 + "\n")

        try:
            while True:
                input("Press Enter to record...")

                # Record
                audio = self.record_audio()

                # Classify
                start_time = time.time()
                predicted_label, confidence, probabilities = self.classify(audio)
                inference_time = (time.time() - start_time) * 1000

                # Display results
                self.print_results(predicted_label, confidence, probabilities)
                print(f"    Inference time: {inference_time:.1f}ms\n")

        except KeyboardInterrupt:
            print("\n\nExiting...")

    def test_audio_file(self, audio_path):
        """Test on a WAV file."""
        import soundfile as sf

        print(f"\nTesting: {audio_path}")
        audio, sr = sf.read(str(audio_path), dtype='float32')

        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Resample if needed
        if sr != self.sample_rate:
            import torchaudio
            audio_t = torch.from_numpy(audio).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio_t).squeeze(0).numpy()

        # Pad or trim to 1 second
        target_length = self.sample_rate
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]

        # Classify
        start_time = time.time()
        predicted_label, confidence, probabilities = self.classify(audio)
        inference_time = (time.time() - start_time) * 1000

        self.print_results(predicted_label, confidence, probabilities)
        print(f"    Inference time: {inference_time:.1f}ms")

        return predicted_label, confidence

    def test_folder(self, folder_path):
        """Test all WAV files in a folder."""
        folder = Path(folder_path)
        wav_files = list(folder.glob("*.wav"))

        if not wav_files:
            print(f"No WAV files found in {folder_path}")
            return

        print(f"\nTesting {len(wav_files)} WAV files from {folder_path}")
        print("=" * 60)

        results = []
        for wav_file in sorted(wav_files):
            label, conf = self.test_audio_file(wav_file)
            results.append((wav_file.name, label, conf))
            print()

        # Summary
        print("=" * 60)
        print("Summary:")
        for name, label, conf in results:
            print(f"  {name:30s} -> {label:10s} ({conf*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Test speech commands model with live audio or WAV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_live_audio.py                          # Interactive mic recording
  python test_live_audio.py --file test.wav          # Test single WAV file
  python test_live_audio.py --folder ./test_wavs/    # Test folder of WAVs
  python test_live_audio.py --list-devices           # List audio devices
        """
    )
    parser.add_argument('--model', type=str, default=str(DEFAULT_MODEL_PATH),
                        help='Path to model checkpoint (default: model_final.pt)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run inference on (cuda/cpu)')
    parser.add_argument('--file', type=str, default=None,
                        help='Test on a single audio file')
    parser.add_argument('--folder', type=str, default=None,
                        help='Test on all WAV files in a folder')
    parser.add_argument('--list-devices', action='store_true',
                        help='List available audio devices')

    args = parser.parse_args()

    # List audio devices if requested
    if args.list_devices:
        import sounddevice as sd
        print("\nAvailable audio devices:")
        print(sd.query_devices())
        return

    # Initialize classifier
    classifier = AudioClassifier(model_path=args.model, device=args.device)

    # Choose mode
    if args.file:
        classifier.test_audio_file(args.file)
    elif args.folder:
        classifier.test_folder(args.folder)
    else:
        classifier.run_interactive()


if __name__ == "__main__":
    main()