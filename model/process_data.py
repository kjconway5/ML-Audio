#!/usr/bin/env python3
"""
Processes all Data from Speech Command Dataset (assumes data/speec_commands_v0.02 directory)
Outputs train,val and test  features amd labels as npy files
"""
import json
import numpy as np
import torch
import torchaudio
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
import random
from pipeline import SimplePipeline

# To Configure
DATA_ROOT = Path(__file__).parent / "data" / "speech_commands_v0.02"
OUTPUT_DIR = Path(__file__).parent / "output"
N_MELS = 64
N_FFT = 512
HOP_LENGTH = 160
USE_FILTERS = True
N_SAMPLES = None

TARGET_KEYWORDS = {"no"}
#Include background noise for silence
INCLUDE_SILENCE = True
#Number of 1 second silence clips
SILENCE_COUNT = 1000
#Set none for no cap
UNKNOWN_MAX_PER_SPLIT = 4000

RANDOM_SEED = 42


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
    val_set = _read_list(root / "validation_list.txt")
    test_set = _read_list(root / "testing_list.txt")

    out = []
    for wav in root.rglob("*.wav"):
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
    return out

#Chop background noise files into 1-second clips and return as Example objects
def generate_silence_examples(root: Path, count: int, target_len: int = 16000):
    
    noise_dir = root / "_background_noise_"
    if not noise_dir.exists():
        print(" _background_noise_ directory not found, skipping silence class")
        return []

    # Load and concatenate all background noise files
    noise_chunks = []
    for wav_path in noise_dir.glob("*.wav"):
        waveform, sr = torchaudio.load(str(wav_path))
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        noise_chunks.append(waveform)

    if not noise_chunks:
        return []

    all_noise = torch.cat(noise_chunks, dim=1)
    total_samples = all_noise.shape[1]

    # Save individual 1 second clips to a temp directory
    silence_dir = root / "_generated_silence_"
    silence_dir.mkdir(exist_ok=True)

    rng = random.Random(RANDOM_SEED)
    examples = []
    for i in range(count):
        start = rng.randint(0, total_samples - target_len)
        clip = all_noise[:, start : start + target_len]
        clip_path = silence_dir / f"silence_{i:05d}.wav"
        torchaudio.save(str(clip_path), clip, 16000)

        # Distribute across splits 
        r = rng.random()
        split = "train" if r < 0.8 else "val" if r < 0.9 else "test"
        examples.append(Example(wav_path=str(clip_path.resolve()), label="silence", split=split))

    return examples

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
    w, in_sr = torchaudio.load(str(wav_path))
    if w.shape[0] > 1:
        w = w.mean(dim=0, keepdim=True)
    if in_sr != sr:
        w = torchaudio.functional.resample(w, in_sr, sr)
    N = int(sr * seconds)
    if w.shape[1] < N:
        w = torch.nn.functional.pad(w, (0, N - w.shape[1]))
    else:
        w = w[:, :N]
    return w.to(torch.float32)


#Converts torch tensor to int16 numpy array
def torch_to_int16_np(waveform: torch.Tensor) -> np.ndarray:
    x = waveform.squeeze(0).numpy().astype(np.float32)
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16)

#Process examples and save as .npy files.
def process_and_save():
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Feature Extractor
    pipeline = SimplePipeline(
        sample_rate=16_000,
        use_filters=USE_FILTERS,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )

    # Collect examples and optionally remap labels
    examples = collect_examples_speech_commands(DATA_ROOT)

    # Optionally add silence class
    if INCLUDE_SILENCE:
        silence_examples = generate_silence_examples(DATA_ROOT, SILENCE_COUNT)
        examples.extend(silence_examples)

    # Optionally subsample "unknown" to reduce imbalance
    if TARGET_KEYWORDS is not None and UNKNOWN_MAX_PER_SPLIT is not None:
        examples = subsample_unknown(examples, UNKNOWN_MAX_PER_SPLIT)

    # Build label mapping
    labels = sorted({ex.label for ex in examples})
    label_to_id = {lab: i for i, lab in enumerate(labels)}

    # Print class summary
    print(f"\nClasses ({len(labels)}): {labels}")
    print(f"Label mapping: {label_to_id}\n")

    # Loop over splits and generate arrays
    for split in ["train", "val", "test"]:
        split_examples = [ex for ex in examples if ex.split == split]
        if N_SAMPLES:
            split_examples = split_examples[:N_SAMPLES]

        # Print per-class counts for this split
        from collections import Counter
        counts = Counter(ex.label for ex in split_examples)
        print(f"{split} class distribution: {dict(sorted(counts.items()))}")

        features_list, labels_list = [], []
        # Process waveforms
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
