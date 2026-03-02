#!/usr/bin/env python3
"""
Processes all Data from Speech Command Dataset.
Reads all settings from config.yaml.
Outputs train, val and test features and labels as npy files.
"""
import json
import yaml
import numpy as np
import torch
import soundfile as sf
from scipy import signal
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
import random
from pipeline import SimplePipeline

print("Using soundfile for audio I/O (Docker-compatible)")

# Load configuration from config.yaml
CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH, 'r') as f:
    _config = yaml.safe_load(f)

# Data settings
DATA_ROOT = Path(_config["dataset"]["data_dir"])
OUTPUT_DIR = Path(_config["data"]["output_dir"])

# Preprocessing settings
_preproc = _config["preprocessing"]
N_MELS = _preproc["n_mels"]
N_FFT = _preproc["n_fft"]
HOP_LENGTH = _preproc["hop_length"]
USE_FILTERS = _preproc["use_filters"]
N_SAMPLES = None

# Data processing settings
_data = _config["data"]
_target_kw_list = _data.get("target_keywords", None)
TARGET_KEYWORDS = set(_target_kw_list) if _target_kw_list else None
INCLUDE_SILENCE = _data.get("include_silence", True)
SILENCE_COUNT = _data.get("num_silence_samples", 1000)
UNKNOWN_MAX_PER_SPLIT = _data.get("unknown_max_per_split", None)
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

#creates data set, assigns labels and splits. Groups non target words to unknowns
def collect_examples_speech_commands(root: Path):
    import glob

    print(f"\n[DEBUG] Starting to collect examples from: {root}")
    val_set = _read_list(root / "validation_list.txt")
    test_set = _read_list(root / "testing_list.txt")
    print(f"[DEBUG] Loaded validation set: {len(val_set)} files, test set: {len(test_set)} files")

    print(f"[DEBUG] Scanning for .wav files (this may take 1-2 minutes)...")
    print(f"[DEBUG] Finding all .wav files...")

    # Use glob.glob which is faster than Path.rglob for large directories
    wav_pattern = str(root / "**" / "*.wav")
    all_wavs = glob.glob(wav_pattern, recursive=True)
    print(f"[DEBUG] Found {len(all_wavs)} total .wav files, now processing...")

    out = []
    for wav_str in tqdm(all_wavs, desc="Collecting examples"):
        wav = Path(wav_str)
        rel = wav.relative_to(root).as_posix()
        # Skip background noise (handled separately) and hidden files
        if rel.startswith("_background_noise_/") or "/." in rel:
            continue
        label = wav.relative_to(root).parts[0]

        # Remap non-target keywords to "unknown"
        if TARGET_KEYWORDS is not None and label not in TARGET_KEYWORDS:
            label = "unknown"

        split = "val" if rel in val_set else "test" if rel in test_set else "train"
        out.append(Example(wav_path=str(wav.resolve()), label=label, split=split))

    print(f"[DEBUG] Collected {len(out)} valid audio files")
    return out

#Chop background noise files into 1-second clips and return as Example objects
def generate_silence_examples(root: Path, count: int, target_len: int = 16000):
    print(f"\n[DEBUG] Generating {count} silence examples from background noise...")
    noise_dir = root / "_background_noise_"
    if not noise_dir.exists():
        print(" _background_noise_ directory not found, skipping silence class")
        return []

    # Load and concatenate all background noise files
    print(f"[DEBUG] Loading background noise files from: {noise_dir}")
    noise_chunks = []
    for wav_path in noise_dir.glob("*.wav"):
        print(f"[DEBUG] Loading: {wav_path.name}")
        waveform, sr = sf.read(str(wav_path), dtype='float32')

        # Convert to mono if stereo
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        # Resample if necessary
        if sr != 16000:
            waveform = resample_audio(waveform, sr, 16000)

        noise_chunks.append(waveform)

    if not noise_chunks:
        return []

    # Concatenate all noise
    all_noise = np.concatenate(noise_chunks)
    total_samples = len(all_noise)

    # Save individual 1 second clips to a temp directory
    silence_dir = root / "_generated_silence_"
    silence_dir.mkdir(exist_ok=True)

    rng = random.Random(RANDOM_SEED)
    examples = []
    for i in range(count):
        start = rng.randint(0, total_samples - target_len)
        clip = all_noise[start : start + target_len]
        clip_path = silence_dir / f"silence_{i:05d}.wav"
        sf.write(str(clip_path), clip, 16000)

        # Distribute across splits 
        r = rng.random()
        split = "train" if r < 0.8 else "val" if r < 0.9 else "test"
        examples.append(Example(wav_path=str(clip_path.resolve()), label="silence", split=split))

    return examples

    #Resample audio using scipy
def resample_audio(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio from orig_sr to target_sr using scipy."""
    if orig_sr == target_sr:
        return waveform
    # Calculate number of samples after resampling
    num_samples = int(len(waveform) * target_sr / orig_sr)
    return signal.resample(waveform, num_samples)

#Cap the number of 'unknown' examples in each split.
def subsample_unknown(examples: list, max_per_split: int) -> list:
    rng = random.Random(RANDOM_SEED)
    kept = []
    for split in ["train", "val", "test"]:
        split_unknown = [ex for ex in examples if ex.split == split and ex.label == "unknown"]
        split_other = [ex for ex in examples if ex.split == split and ex.label != "unknown"]
        if len(split_unknown) > max_per_split:
            split_unknown = rng.sample(split_unknown, max_per_split)
        kept.extend(split_other + split_unknown)
    return kept

#Adjusts input wav to ensure they match our sample rate and duration
def load_wav_fixed(wav_path: Path, sr: int = 16_000, seconds: float = 1.0) -> torch.Tensor:
    # Load audio with soundfile
    w, in_sr = sf.read(str(wav_path), dtype='float32')

    # Convert to mono if stereo (average channels)
    if w.ndim > 1:
        w = w.mean(axis=1)

    # Resample if necessary
    if in_sr != sr:
        w = resample_audio(w, in_sr, sr)

    # Ensure correct length
    N = int(sr * seconds)
    if len(w) < N:
        # Pad with zeros
        w = np.pad(w, (0, N - len(w)), mode='constant')
    else:
        # Truncate
        w = w[:N]

    # Convert to torch tensor with shape (1, N) to match original format
    return torch.from_numpy(w).unsqueeze(0).float()


#Converts torch tensor to int16 numpy array
def torch_to_int16_np(waveform: torch.Tensor) -> np.ndarray:
    x = waveform.squeeze(0).numpy().astype(np.float32)
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16)

#Process examples and save as .npy files.
def process_and_save():
    print("\n" + "="*60)
    print("STARTING DATA PROCESSING")
    print("="*60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[DEBUG] Output directory: {OUTPUT_DIR}")

    # Feature Extractor
    print(f"[DEBUG] Initializing pipeline with: n_mels={N_MELS}, n_fft={N_FFT}, hop_length={HOP_LENGTH}")
    pipeline = SimplePipeline(
        sample_rate=_preproc.get("sample_rate", 16_000),
        use_filters=USE_FILTERS,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window_length=_preproc.get("window_length", N_FFT),
        hpf_order=_preproc.get("hpf_order", 2),
        lpf_order=_preproc.get("lpf_order", 4),
        cutoff_hpf=_preproc.get("cutoff_hpf", 150),
        cutoff_lpf=_preproc.get("cutoff_lpf", 4000),
    )
    print(f"[DEBUG] Pipeline initialized successfully")

    # Collect examples and optionally remap labels
    print(f"\n[DEBUG] Target keywords: {TARGET_KEYWORDS}")
    examples = collect_examples_speech_commands(DATA_ROOT)

    # Optionally add silence class
    if INCLUDE_SILENCE:
        print(f"[DEBUG] Include silence is enabled, generating {SILENCE_COUNT} silence samples...")
        silence_examples = generate_silence_examples(DATA_ROOT, SILENCE_COUNT)
        examples.extend(silence_examples)
        print(f"[DEBUG] Total examples after adding silence: {len(examples)}")
    else:
        print(f"[DEBUG] Silence generation is disabled")

    # Optionally subsample "unknown" to reduce imbalance
    if TARGET_KEYWORDS is not None and UNKNOWN_MAX_PER_SPLIT is not None:
        print(f"[DEBUG] Subsampling unknown class to max {UNKNOWN_MAX_PER_SPLIT} per split...")
        examples = subsample_unknown(examples, UNKNOWN_MAX_PER_SPLIT)
        print(f"[DEBUG] Total examples after subsampling: {len(examples)}")

    # Build label mapping
    labels = sorted({ex.label for ex in examples})
    label_to_id = {lab: i for i, lab in enumerate(labels)}

    # Print class summary
    print(f"\nClasses ({len(labels)}): {labels}")
    print(f"Label mapping: {label_to_id}\n")

    # Loop over splits and generate arrays
    for split in ["train", "val", "test"]:
        print(f"\n{'='*60}")
        print(f"PROCESSING {split.upper()} SPLIT")
        print(f"{'='*60}")

        split_examples = [ex for ex in examples if ex.split == split]
        if N_SAMPLES:
            split_examples = split_examples[:N_SAMPLES]

        print(f"[DEBUG] {split} split has {len(split_examples)} examples")

        # Print per-class counts for this split
        from collections import Counter
        counts = Counter(ex.label for ex in split_examples)
        print(f"{split} class distribution: {dict(sorted(counts.items()))}")

        features_list, labels_list = [], []
        # Process waveforms
        print(f"[DEBUG] Starting audio processing for {split} split...")
        for ex in tqdm(split_examples, desc=f"Processing {split}"):
            wav = load_wav_fixed(Path(ex.wav_path))
            audio_i16 = torch_to_int16_np(wav)
            feats, _ = pipeline.process(audio_i16)
            features_list.append(feats.T)  # (T, M)
            labels_list.append(label_to_id[ex.label])

        # Stack into array
        features = np.stack(features_list, axis=0).astype(np.float32)
        labels_arr = np.array(labels_list, dtype=np.int64)

        np.save(OUTPUT_DIR / f"{split}_features.npy", features)
        np.save(OUTPUT_DIR / f"{split}_labels.npy", labels_arr)
        print(f"Saved {split}: features {features.shape}, labels {labels_arr.shape}\n")

    # Save config
    config = {
        "labels": labels,
        "label_to_id": label_to_id,
        "target_keywords": sorted(TARGET_KEYWORDS) if TARGET_KEYWORDS else None,
        "include_silence": INCLUDE_SILENCE,
        "pipeline": pipeline.get_config(),
    }
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("Saved config.json")


if __name__ == "__main__":
    process_and_save()