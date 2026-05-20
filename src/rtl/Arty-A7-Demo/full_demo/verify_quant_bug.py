#!/usr/bin/env python3
"""Definitive test for the suspected INPUT_QUANT_MULT mismatch.

input_quant.txt says the model was trained/exported with mult=5795362,
but pipeline_top.sv has INPUT_QUANT_MULT=5817845 baked in. Run the
PyTorch DSCNN three ways on yes.wav and compare:

  A) re-quant with correct mult (5795362) → expected: yes
  B) re-quant with FPGA mult (5817845)    → expected: matches FPGA's wrong answer (unknown)
  C) use the actual FPGA-captured spect bank as-is → matches B

If B matches the FPGA result and A doesn't, the input_quant_mult bug is confirmed.
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO / "src" / "ml"))
sys.path.insert(0, str(_REPO / "src" / "ml" / "my_test"))

from golden_model import GoldenExtractor   # noqa: E402

# Monkey-patch torch.quantization.get_default_qat_qconfig to accept the
# checkpoint's "qnnpack_int8sym" tag (not in this torch version) by
# falling back to fbgemm — only the int8 weights matter for inference here.
import torch.quantization as _tq          # noqa: E402
_real_get_qat = _tq.get_default_qat_qconfig
def _patched_get_qat(backend="fbgemm"):
    if backend not in {"fbgemm", "x86", "qnnpack", "onednn"}:
        backend = "fbgemm"
    return _real_get_qat(backend)
_tq.get_default_qat_qconfig = _patched_get_qat

from run_rtl_wav import load_model         # noqa: E402

import wave

WAV_PATH    = _REPO / "src" / "ml" / "my_test" / "test_wavs" / "yes.wav"
MODEL_PATH  = _REPO / "src" / "ml" / "models" / "dscnn-24center-v1" / "dscnn-24center-v1.pt"
FPGA_CAPTURE= Path(__file__).parent / "captures_yes" / "spect_bank_a.npy"

START_FRAME = 37
N_FRAMES    = 50
N_MELS      = 40
MULT_MODEL  = 5795362   # what the model was exported with (input_quant.txt)
MULT_FPGA   = 5817845   # what pipeline_top.sv has baked in (the suspected bug)
SHIFT       = 31


def read_wav(p: Path) -> np.ndarray:
    with wave.open(str(p), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")


def requant(q610: np.ndarray, mult: int) -> np.ndarray:
    """Mirror spect_buffer_ctrl.sv USE_INPUT_REQUANT=1 path with arbitrary mult."""
    prod = q610.astype(np.int64) * mult
    out  = (prod + (1 << (SHIFT - 1))) >> SHIFT
    return np.clip(out, -128, 127).astype(np.int8)


def classify(model, spect_int8: np.ndarray, labels):
    # model expects (1, 1, n_frames, n_mels) float
    t = torch.from_numpy(spect_int8.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits = model(t)
        probs  = torch.softmax(logits, dim=1)
        idx    = int(probs.argmax(dim=1).item())
    print(f"    logits   : {[f'{labels[i]}={logits[0,i].item():+.2f}' for i in range(len(labels))]}")
    print(f"    pred     : {labels[idx]}  (confidence {probs[0,idx].item()*100:.1f}%)")
    return idx


def main():
    print(f"Loading model: {MODEL_PATH}")
    model, labels = load_model(MODEL_PATH)
    print(f"  classes: {labels}")

    audio = read_wav(WAV_PATH)
    golden_uint16 = GoldenExtractor(bfp_compensate=True).extract(audio)   # (40, n_frames) Q6.10
    window = golden_uint16[:, START_FRAME : START_FRAME + N_FRAMES].T     # (50, 40) uint16

    print(f"\n--- A) re-quant with CORRECT mult={MULT_MODEL} (what the model expects) ---")
    spect_A = requant(window, MULT_MODEL)
    classify(model, spect_A, labels)

    print(f"\n--- B) re-quant with FPGA mult={MULT_FPGA} (the suspected wrong value) ---")
    spect_B = requant(window, MULT_FPGA)
    classify(model, spect_B, labels)

    print(f"\n--- C) the actual FPGA-captured spect bank (should agree with B) ---")
    spect_C = np.load(FPGA_CAPTURE).astype(np.int8)
    classify(model, spect_C, labels)

    # Cross-check: how different are B and C? (should be near-identical if mult is the only diff)
    diff_BC = np.abs(spect_B.astype(np.int32) - spect_C.astype(np.int32))
    print(f"\nspect_B (Python wrong-mult) vs spect_C (FPGA): max|diff|={diff_BC.max()}, mean={diff_BC.mean():.2f}")


if __name__ == "__main__":
    sys.exit(main())
