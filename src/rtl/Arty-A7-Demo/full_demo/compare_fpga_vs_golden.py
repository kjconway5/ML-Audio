#!/usr/bin/env python3
"""Compare an FPGA-captured spectrogram bank (via --read-spect) against
the bit-accurate golden_model.GoldenExtractor running on the same WAV.

Usage:
  # 1. boot + classify (also leaves the spect bank populated):
  python3 host/load_full.py -p /dev/ttyUSB1 -b 460800 --classify <wav> \\
      --weights <model>/weights.hex --bias <model>/bias.hex --cfg host/cfg.hex
  # 2. dump the bank:
  python3 host/load_full.py -p /dev/ttyUSB1 -b 460800 --read-spect a \\
      --spect-out captures/
  # 3. compare:
  python3 compare_fpga_vs_golden.py <wav> captures/spect_bank_a.npy

If golden and FPGA match (or differ only in the int8-quant LSB), the
features pipeline is bit-correct and any classification mismatch must
be downstream (weights/bias/cfg loading into DSCNN).
"""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np

# Import the canonical golden extractor (pipeline_top.sv replica)
# This file: src/rtl/Arty-A7-Demo/full_demo/  →  parents[4] = repo root
_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO / "src" / "ml"))
from golden_model import GoldenExtractor  # noqa: E402

# Constants must match spect_buffer_ctrl.sv defaults used by pipeline_top
START_FRAME      = 37
N_FRAMES         = 50
N_MELS           = 40
INPUT_QUANT_MULT = 5817845
INPUT_QUANT_SHIFT= 31


def read_wav_int16(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2 and w.getframerate() == 16000
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")


def requant_q610_to_int8(q610: np.ndarray) -> np.ndarray:
    """Replicate spect_buffer_ctrl.sv's int8 quantization (USE_INPUT_REQUANT=1)."""
    prod  = q610.astype(np.int64) * INPUT_QUANT_MULT
    round_bias = 1 << (INPUT_QUANT_SHIFT - 1)
    out = (prod + round_bias) >> INPUT_QUANT_SHIFT
    return np.clip(out, -128, 127).astype(np.int8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav",  type=Path)
    ap.add_argument("bank", type=Path, help="captures/spect_bank_a.npy or _b.npy from --read-spect")
    ap.add_argument("--save-golden", type=Path, default=None,
                    help="also save golden int8 spectrogram here for visual diff")
    args = ap.parse_args()

    fpga = np.load(args.bank)  # (50, 40) int8
    assert fpga.shape == (N_FRAMES, N_MELS), f"unexpected FPGA shape {fpga.shape}"

    audio   = read_wav_int16(args.wav)
    golden  = GoldenExtractor(bfp_compensate=True).extract(audio)   # (40, n_frames) uint16 Q6.10
    n_total = golden.shape[1]
    if n_total < START_FRAME + N_FRAMES:
        print(f"WARN: golden produced {n_total} frames, need {START_FRAME + N_FRAMES}; padding")
        pad = np.zeros((N_MELS, START_FRAME + N_FRAMES - n_total), dtype=golden.dtype)
        golden = np.concatenate([golden, pad], axis=1)

    # Trim to the 50-frame window the FPGA captures, transpose to (50, 40),
    # then apply the same int8 requant the hardware does.
    golden_window = golden[:, START_FRAME : START_FRAME + N_FRAMES].T   # (50, 40) uint16
    golden_int8   = requant_q610_to_int8(golden_window)                  # (50, 40) int8

    if args.save_golden is not None:
        args.save_golden.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_golden, golden_int8)
        print(f"  saved golden int8 → {args.save_golden}")

    diff       = fpga.astype(np.int32) - golden_int8.astype(np.int32)
    exact_eq   = int((diff == 0).sum())
    within_1   = int((np.abs(diff) <= 1).sum())
    within_4   = int((np.abs(diff) <= 4).sum())
    max_abs    = int(np.abs(diff).max())
    total      = diff.size

    print()
    print(f"FPGA  bank: shape={fpga.shape},  range=[{fpga.min():4d},{fpga.max():4d}], mean={fpga.mean():.1f}")
    print(f"Golden int: shape={golden_int8.shape}, range=[{golden_int8.min():4d},{golden_int8.max():4d}], mean={golden_int8.mean():.1f}")
    print()
    print(f"  exact match: {exact_eq:>5}/{total} ({100*exact_eq/total:5.1f}%)")
    print(f"  |diff| ≤ 1 : {within_1:>5}/{total} ({100*within_1/total:5.1f}%)")
    print(f"  |diff| ≤ 4 : {within_4:>5}/{total} ({100*within_4/total:5.1f}%)")
    print(f"  max |diff| : {max_abs}")

    fpga_argmax   = np.argmax(fpga,        axis=1)
    golden_argmax = np.argmax(golden_int8, axis=1)
    argmax_agree  = int((fpga_argmax == golden_argmax).sum())
    print(f"  argmax-per-frame agreement: {argmax_agree}/{N_FRAMES} frames")
    print(f"    FPGA   argmax (first 20): {fpga_argmax[:20].tolist()}")
    print(f"    Golden argmax (first 20): {golden_argmax[:20].tolist()}")

    # Pearson correlation as a coarse shape-match metric (de-means per frame)
    fpga_f   = fpga.astype(np.float64).ravel()
    golden_f = golden_int8.astype(np.float64).ravel()
    corr     = np.corrcoef(fpga_f, golden_f)[0, 1]
    print(f"  flat correlation: {corr:.4f}  (1.0 = perfect, 0 = unrelated)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
