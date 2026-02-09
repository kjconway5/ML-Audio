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

from pipeline import SimplePipeline

# To Configure
DATA_ROOT = Path(__file__).parent / "data" / "speech_commands_v0.02"
OUTPUT_DIR = Path(__file__).parent / "output"
N_MELS = 64
N_FFT = 512
HOP_LENGTH = 160
USE_FILTERS = True
N_SAMPLES = None


@dataclass(frozen=True)
class Example:
    wav_path: str
    label: str
    split: str


def _read_list(p: Path) -> set:
    if not p.exists():
        return set()
    return set(line.strip().replace("\\", "/") for line in p.read_text().splitlines() if line.strip())


#creates data set indexer
def collect_examples_speech_commands(root: Path):
    val_set = _read_list(root / "validation_list.txt")
    test_set = _read_list(root / "testing_list.txt")

    out = []
    for wav in root.rglob("*.wav"):
        rel = wav.relative_to(root).as_posix()
        if rel.startswith("_background_noise_/") or "/." in rel:
            continue
        label = wav.relative_to(root).parts[0]
        split = "val" if rel in val_set else "test" if rel in test_set else "train"
        out.append(Example(wav_path=str(wav.resolve()), label=label, split=split))
    return out


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


def process_and_save():
    #Process examples and save as .npy files.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    #Feature Extractor
    pipeline = SimplePipeline(
        sample_rate=16_000,
        use_filters=USE_FILTERS,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    #Collect examples and build label mapping
    examples = collect_examples_speech_commands(DATA_ROOT)
    labels = sorted({ex.label for ex in examples})
    label_to_id = {lab: i for i, lab in enumerate(labels)}
    #loop over splits and generate arrays
    for split in ["train", "val", "test"]:
        split_examples = [ex for ex in examples if ex.split == split]
        if N_SAMPLES:
            split_examples = split_examples[:N_SAMPLES]

        features_list, labels_list = [], []
        #process waveforms
        for ex in tqdm(split_examples, desc=f"Processing {split}"):
            wav = load_wav_fixed(Path(ex.wav_path))
            audio_i16 = torch_to_int16_np(wav)
            feats, _ = pipeline.process(audio_i16)
            features_list.append(feats.T)  # (T, M)
            labels_list.append(label_to_id[ex.label])
        #stack into array
        features = np.stack(features_list, axis=0).astype(np.float32)
        labels_arr = np.array(labels_list, dtype=np.int64)

        np.save(OUTPUT_DIR / f"{split}_features.npy", features)
        np.save(OUTPUT_DIR / f"{split}_labels.npy", labels_arr)
        print(f"Saved {split}: features {features.shape}, labels {labels_arr.shape}")

    # Save config
    config = {
        "labels": labels,
        "label_to_id": label_to_id,
        "pipeline": pipeline.get_config(),
    }
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config.json")


if __name__ == "__main__":
    process_and_save()
