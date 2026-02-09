#!/usr/bin/env python3
"""
Speech Commands -> wav -> YOUR SimplePipeline (log-mel) -> PyTorch Dataset/DataLoader (single file).

Assumptions:
- You have pipeline.py (and its deps) importable.
- SimplePipeline.process(audio_int16) returns features shaped (n_mels, n_frames).
- We standardize audio to 16kHz mono 1.0s, convert to np.int16, then call your pipeline.

Dataset returns:
  x: (1, T, M) float32   (channel, time, mel)
  y: () int64
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

from pipeline import SimplePipeline


# ----------------------------- #
# Dataset indexing / splits
# ----------------------------- #

@dataclass(frozen=True)
class Example:
    wav_path: str
    label: str
    split: str  # "train" | "val" | "test"


def _read_list(p: Path) -> set[str]:
    if not p.exists():
        return set()
    return set(line.strip().replace("\\", "/") for line in p.read_text().splitlines() if line.strip())


def collect_examples_speech_commands(root: Path) -> List[Example]:
    val_set = _read_list(root / "validation_list.txt")
    test_set = _read_list(root / "testing_list.txt")

    out: List[Example] = []
    for wav in root.rglob("*.wav"):
        rel = wav.relative_to(root).as_posix()
        if rel.startswith("_background_noise_/") or "/." in rel:
            continue

        label = wav.relative_to(root).parts[0]
        split = "val" if rel in val_set else "test" if rel in test_set else "train"
        out.append(Example(wav_path=str(wav.resolve()), label=label, split=split))
    return out


# ----------------------------- #
# Audio standardization
# ----------------------------- #

def load_wav_fixed(
    wav_path: Path, *, sr: int = 16_000, seconds: float = 1.0
) -> torch.Tensor:
    """Return waveform (1, N) float32, mono, resampled, padded/cropped to fixed duration."""
    w, in_sr = torchaudio.load(str(wav_path))  # (C, N)
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


def torch_to_int16_np(waveform: torch.Tensor) -> np.ndarray:
    """(1, N) float32 in [-1,1] -> (N,) np.int16"""
    x = waveform.squeeze(0).detach().cpu().numpy().astype(np.float32)
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16)


# ----------------------------- #
# PyTorch Dataset
# ----------------------------- #

class SpeechCommandsLogMelDataset(Dataset):
    def __init__(
        self,
        examples: List[Example],
        label_to_id: Dict[str, int],
        pipeline: SimplePipeline,
        *,
        split: Optional[str] = None,
        sr: int = 16_000,
        seconds: float = 1.0,
    ):
        self.examples = [ex for ex in examples if split is None or ex.split == split]
        self.label_to_id = label_to_id
        self.pipeline = pipeline
        self.sr = sr
        self.seconds = seconds

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ex = self.examples[idx]
        w = load_wav_fixed(Path(ex.wav_path), sr=self.sr, seconds=self.seconds)
        audio_i16 = torch_to_int16_np(w)

        # Your pipeline: (n_mels, n_frames)
        feats, _ = self.pipeline.process(audio_i16)

        feats = torch.from_numpy(np.asarray(feats)).to(torch.float32)  # (M, T)
        x = feats.transpose(0, 1).unsqueeze(0).contiguous()           # (1, T, M)
        y = torch.tensor(self.label_to_id[ex.label], dtype=torch.long)
        return x, y


# ----------------------------- #
# Main sanity check
# ----------------------------- #

def main():
    dataset_root = Path(__file__).parent / "data" / "speech_commands_v0.02"

    examples = collect_examples_speech_commands(dataset_root)
    labels = sorted({ex.label for ex in examples})
    label_to_id = {lab: i for i, lab in enumerate(labels)}

    pipe = SimplePipeline(sample_rate=16_000, use_filters=True, n_mels=64, n_fft=512, hop_length=160)

    train_ds = SpeechCommandsLogMelDataset(examples, label_to_id, pipe, split="train")
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)

    x, y = next(iter(train_loader))
    print("x:", x.shape, x.dtype)  # (B, 1, T, M)
    print("y:", y.shape, y.dtype)  # (B,)


if __name__ == "__main__":
    main()
