#!/usr/bin/env python3
"""
Processes Speech Command Dataset using FullPipelineGoldenExtractor (RTL-accurate features).

This replaces process_data.py (which uses torchaudio LogMelExtractor) to ensure
training features are identical to what the RTL hardware produces, including the
full front-end: sigma-delta PDM, CIC decimator, and compensation FIR.

Key differences from process_data.py:
  1. Uses FullPipelineGoldenExtractor (PCM -> PDM -> CIC -> compFIR -> STFFT)
  2. Scales audio to 14-bit signed range (±8191) to match RTL ADC dynamic range
  3. No center padding (matches RTL: no zero-padding at signal edges)
  4. Crops/pads to frames 37..86, matching spect_buffer_ctrl.sv START_FRAME
  5. Fixed-point mel filterbank, log LUT, and power calculation

Usage:
    python process_full_data.py                    # uses config.yaml
    python process_full_data.py --config dscnn/config.yaml
"""
import json
import multiprocessing as mp
import yaml
import numpy as np
import torch
import soundfile as sf
from scipy import signal
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
import random
import sys

# Add parent directory to import golden model
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from golden_model import FullPipelineGoldenExtractor

print("Using FullPipelineGoldenExtractor (PDM -> CIC -> compFIR -> STFFT)")

# Load configuration from config.yaml
CONFIG_PATH = _HERE / "config.yaml"
with open(CONFIG_PATH, 'r') as f:
    _config = yaml.safe_load(f)

# Data settings
DATA_ROOT = Path(_config["dataset"]["data_dir"])
OUTPUT_DIR = Path(_config["data"]["output_dir"])

# Preprocessing settings
_preproc = _config["preprocessing"]
SAMPLE_RATE = _preproc.get("sample_rate", 16000)
N_MELS = _preproc["n_mels"]
N_FFT = _preproc["n_fft"]
HOP_LENGTH = _preproc["hop_length"]
USE_FILTERS = _preproc.get("use_filters", False)
N_FRAMES = 50
START_FRAME = 37

# RTL ADC range: 14-bit signed
SAMPLE_W = 14
SAMPLE_MAX = (1 << (SAMPLE_W - 1)) - 1  # 8191

# Data processing settings
_data = _config["data"]
_target_kw_list = _data.get("target_keywords", None)
TARGET_KEYWORDS = set(_target_kw_list) if _target_kw_list else None
INCLUDE_SILENCE = _data.get("include_silence", True)
SILENCE_COUNT = _data.get("num_silence_samples", 1000)
UNKNOWN_MAX = {
    "train": _data.get("unknown_max_train", None),
    "val":   _data.get("unknown_max_val",   None),
    "test":  _data.get("unknown_max_test",  None),
}
RANDOM_SEED = _data.get("random_seed", 42)


@dataclass(frozen=True)
class Example:
    wav_path: str
    label: str
    split: str


def _read_list(p: Path) -> set:
    if not p.exists():
        return set()
    return set(
        line.strip().replace("\\", "/")
        for line in p.read_text().splitlines()
        if line.strip()
    )


def collect_examples_speech_commands(root: Path):
    import glob

    print(f"\n[DEBUG] Collecting examples from: {root}")
    val_set = _read_list(root / "validation_list.txt")
    test_set = _read_list(root / "testing_list.txt")

    wav_pattern = str(root / "**" / "*.wav")
    all_wavs = glob.glob(wav_pattern, recursive=True)
    print(f"[DEBUG] Found {len(all_wavs)} total .wav files")

    out = []
    for wav_str in tqdm(all_wavs, desc="Collecting examples"):
        wav = Path(wav_str)
        rel = wav.relative_to(root).as_posix()
        if rel.startswith("_background_noise_/") or "/." in rel:
            continue
        label = wav.relative_to(root).parts[0]
        if TARGET_KEYWORDS is not None and label not in TARGET_KEYWORDS:
            label = "unknown"
        split = "val" if rel in val_set else "test" if rel in test_set else "train"
        out.append(Example(wav_path=str(wav.resolve()), label=label, split=split))

    return out


def generate_silence_examples(root: Path, count: int, target_len: int = 16000):
    noise_dir = root / "_background_noise_"
    if not noise_dir.exists():
        print(" _background_noise_ directory not found, skipping silence")
        return []

    noise_chunks = []
    for wav_path in noise_dir.glob("*.wav"):
        waveform, sr = sf.read(str(wav_path), dtype='float32')
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if sr != SAMPLE_RATE:
            waveform = resample_audio(waveform, sr, SAMPLE_RATE)
        noise_chunks.append(waveform)

    if not noise_chunks:
        return []

    all_noise = np.concatenate(noise_chunks)
    total_samples = len(all_noise)

    silence_dir = root / "_generated_silence_"
    silence_dir.mkdir(exist_ok=True)

    rng = random.Random(RANDOM_SEED)
    examples = []
    for i in range(count):
        start = rng.randint(0, total_samples - target_len)
        clip = all_noise[start : start + target_len]
        clip_path = silence_dir / f"silence_{i:05d}.wav"
        sf.write(str(clip_path), clip, SAMPLE_RATE)
        r = rng.random()
        split = "train" if r < 0.8 else "val" if r < 0.9 else "test"
        examples.append(Example(wav_path=str(clip_path.resolve()), label="silence", split=split))

    return examples


def resample_audio(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return waveform
    num_samples = int(len(waveform) * target_sr / orig_sr)
    return signal.resample(waveform, num_samples)


def subsample_unknown(examples: list, max_per_split: dict) -> list:
    rng = random.Random(RANDOM_SEED)
    kept = []
    for split in ["train", "val", "test"]:
        cap = max_per_split.get(split)
        split_unknown = [ex for ex in examples if ex.split == split and ex.label == "unknown"]
        split_other   = [ex for ex in examples if ex.split == split and ex.label != "unknown"]
        if cap is not None and len(split_unknown) > cap:
            split_unknown = rng.sample(split_unknown, cap)
        kept.extend(split_other + split_unknown)
    return kept


def load_wav_fixed(wav_path: Path, sr: int = 16000, seconds: float = 1.0) -> np.ndarray:
    """Load WAV, resample, pad/trim to target duration. Returns float32 [-1, 1]."""
    w, in_sr = sf.read(str(wav_path), dtype='float32')
    if w.ndim > 1:
        w = w.mean(axis=1)
    if in_sr != sr:
        w = resample_audio(w, in_sr, sr)
    N = int(sr * seconds)
    if len(w) < N:
        w = np.pad(w, (0, N - len(w)), mode='constant')
    else:
        w = w[:N]
    return w


def float_to_int14(audio_float: np.ndarray) -> np.ndarray:
    """Convert float32 [-1, 1] audio to 14-bit signed integer range (±8191).

    This matches the RTL ADC dynamic range. Using int16 (±32767) would cause
    the Q4.12 log output to saturate for nearly all mel bins.
    """
    clipped = np.clip(audio_float, -1.0, 1.0)
    return (clipped * SAMPLE_MAX).astype(np.int16)


_worker_extractor = None

def _worker_init():
    global _worker_extractor
    _worker_extractor = FullPipelineGoldenExtractor()

def crop_or_pad_frames(
    feats: np.ndarray,
    n_frames: int = N_FRAMES,
    start_frame: int = START_FRAME,
) -> np.ndarray:
    """Return frame-major features matching RTL's selected fixed spectrogram window."""
    feats_T = feats.T.astype(np.float32)  # (frames, n_mels)
    n = feats_T.shape[0]
    if n > start_frame:
        window = feats_T[start_frame:start_frame + n_frames, :]
    else:
        window = feats_T[:0, :]
    if window.shape[0] >= n_frames:
        return window[:n_frames, :]
    return np.pad(window, ((0, n_frames - window.shape[0]), (0, 0)), mode="constant")


def _process_one(args):
    wav_path, label_id = args
    try:
        audio_float = load_wav_fixed(Path(wav_path))
        audio_i14 = float_to_int14(audio_float)
        feats = _worker_extractor.extract_float(audio_i14)
        n_sat = float((feats >= 15.999).sum() / max(feats.size, 1))
        return crop_or_pad_frames(feats), int(label_id), n_sat
    except Exception:
        return None


def process_and_save():
    print("\n" + "="*60)
    print("GOLDEN MODEL DATA PROCESSING")
    print("="*60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Print config from a temporary instance (workers each create their own)
    _tmp = FullPipelineGoldenExtractor()
    print(f"FullPipelineGoldenExtractor config: {_tmp._stfft.get_config()}")
    print(f"Audio scaling: float32 [-1,1] -> int14 [±{SAMPLE_MAX}]")

    # Verify frame count for 1-second audio
    test_audio = np.zeros(SAMPLE_RATE, dtype=np.int16)
    test_feats = _tmp.extract_float(test_audio)
    raw_frames = test_feats.shape[1]
    print(f"Raw frontend frames for 1-second audio: {raw_frames}")
    print(f"Training/inference frames after RTL crop/pad: {N_FRAMES}")
    print(f"RTL spectrogram start frame: {START_FRAME}")

    # Collect examples
    examples = collect_examples_speech_commands(DATA_ROOT)
    if INCLUDE_SILENCE:
        silence_examples = generate_silence_examples(DATA_ROOT, SILENCE_COUNT)
        examples.extend(silence_examples)
    if TARGET_KEYWORDS is not None and any(v is not None for v in UNKNOWN_MAX.values()):
        examples = subsample_unknown(examples, UNKNOWN_MAX)

    # Build label mapping (alphabetical)
    labels = sorted({ex.label for ex in examples})
    label_to_id = {lab: i for i, lab in enumerate(labels)}
    print(f"\nClasses ({len(labels)}): {labels}")
    print(f"Label mapping: {label_to_id}\n")

    for split in ["train", "val", "test"]:
        print(f"\n{'='*60}")
        print(f"PROCESSING {split.upper()} SPLIT")
        print(f"{'='*60}")

        split_examples = [ex for ex in examples if ex.split == split]
        print(f"{split}: {len(split_examples)} examples")

        from collections import Counter
        counts = Counter(ex.label for ex in split_examples)
        print(f"Class distribution: {dict(sorted(counts.items()))}")

        features_list, labels_list = [], []
        skipped = 0
        saturation_stats = []

        n_workers = mp.cpu_count()
        args = [(ex.wav_path, label_to_id[ex.label]) for ex in split_examples]
        with mp.Pool(n_workers, initializer=_worker_init) as pool:
            for result in tqdm(pool.imap(_process_one, args, chunksize=32),
                               total=len(split_examples), desc=f"Processing {split}"):
                if result is None:
                    skipped += 1
                    continue
                feats_T, label_id, sat = result
                features_list.append(feats_T)
                labels_list.append(label_id)
                saturation_stats.append(sat)

        if skipped > 0:
            print(f"Skipped {skipped} corrupted/invalid files")

        features = np.stack(features_list, axis=0).astype(np.float32)
        labels_arr = np.array(labels_list, dtype=np.int64)

        np.save(OUTPUT_DIR / f"{split}_features.npy", features)
        np.save(OUTPUT_DIR / f"{split}_labels.npy", labels_arr)
        print(f"Saved {split}: features {features.shape}, labels {labels_arr.shape}")
        print(f"  Feature range: [{features.min():.4f}, {features.max():.4f}]")
        print(f"  Feature mean:  {features.mean():.4f}, std: {features.std():.4f}")
        if saturation_stats:
            mean_sat = np.mean(saturation_stats)
            print(f"  Avg saturation: {mean_sat*100:.1f}% of bins per sample")

    # Save config
    config = {
        "labels": labels,
        "label_to_id": label_to_id,
        "target_keywords": sorted(TARGET_KEYWORDS) if TARGET_KEYWORDS else None,
        "include_silence": INCLUDE_SILENCE,
        "feature_extractor": "FullPipelineGoldenExtractor",
        "audio_scaling": f"float32 [-1,1] -> int14 [±{SAMPLE_MAX}]",
        "frame_window": "fixed_start",
        "start_frame": START_FRAME,
        "n_frames": N_FRAMES,
        "pipeline": _tmp._stfft.get_config(),
    }
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("\nSaved config.json")
    print(f"\nTo retrain: python train.py --config {CONFIG_PATH}")


if __name__ == "__main__":
    process_and_save()
