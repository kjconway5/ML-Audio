"""
Bit-accurate Python replica of the RTL feature pipeline.
stfft.sv → power_calc.sv → mel_filterbank.sv → log_lut.sv

Output: np.uint16 Q4.12 log-mel features (same format the CNN sees from RTL).
Use extract_float() for float32 in log₂ units.
"""

import numpy as np
import torch
import os
from pathlib import Path

# RTL parameters (must match logmel_top defaults)
SAMPLE_RATE = 16000
N_FFT       = 256
N_BINS      = N_FFT // 2 + 1   # 129
N_MELS      = 40
WIN_LENGTH  = 256
HOP_LENGTH  = N_FFT // 2       # 128

SAMPLE_W    = 14
FFT_W       = 18
SHIFT       = 6
POWER_W     = 2 * FFT_W - SHIFT + 1  # 31
WEIGHT_W    = 16                # Q0.15 mel coefficients
FRAC_BITS   = 15
ACCUM_W     = 54
MAX_COEFFS  = 16
LOG_OUT_W   = 16
LUT_FRAC    = 6
Q_FRAC      = 12

POWER_MASK = (1 << POWER_W) - 1
ACCUM_MASK = (1 << ACCUM_W) - 1
LOG_MASK   = (1 << LOG_OUT_W) - 1
SAMPLE_MAX = (1 << (SAMPLE_W - 1)) - 1   # 8191

WIN_COEFF_SCALE = 2047

# windowfn.v rounding parameters
_IW = SAMPLE_W
_TW = SAMPLE_W
_OW = SAMPLE_W
_AW = _IW + _TW  # 28

# Hex data paths
_HERE       = Path(__file__).resolve().parent
_DATA_DIR   = _HERE.parent / "rtl" / "Log-Mel" / "data"
_WIN_HEX    = _HERE.parent / "rtl" / "STFFT" / "hanning.hex"


def _load_hex(path: Path) -> list[int]:
    with open(path) as f:
        return [int(line.strip(), 16) for line in f if line.strip()]


class GoldenExtractor:
    """Bit-accurate replica of the RTL STFFT → Log-Mel pipeline."""

    def __init__(self):
        self.win_coeffs = np.array(_load_hex(_WIN_HEX), dtype=np.int32)
        assert len(self.win_coeffs) == N_FFT

        # Mel filterbank: sparse ROM → dense matrix
        raw_coeffs = _load_hex(_DATA_DIR / "mel_coeffs.hex")
        self.mel_starts = _load_hex(_DATA_DIR / "mel_starts.hex")
        self.mel_ends   = _load_hex(_DATA_DIR / "mel_ends.hex")
        self.mel_coeffs = np.array(raw_coeffs, dtype=np.int64).reshape(N_MELS, MAX_COEFFS)

        self.fb_dense = np.zeros((N_BINS, N_MELS), dtype=np.int64)
        for m in range(N_MELS):
            s = self.mel_starts[m]
            e = self.mel_ends[m]
            self.fb_dense[s:e+1, m] = self.mel_coeffs[m, :e - s + 1]

        # Log2 LUT (Q4.12)
        self.log_lut = _load_hex(_DATA_DIR / "log2_lut.hex")
        assert len(self.log_lut) == (1 << LUT_FRAC)

    def _window_frame(self, frame: np.ndarray) -> np.ndarray:
        """windowfn.v: signed multiply + convergent rounding, IW=OW=TW=14."""
        data = frame.astype(np.int64)
        tap  = self.win_coeffs.astype(np.int64)
        product = data * tap

        shift_amt = _AW - _OW  # 14
        round_bit = (product >> shift_amt) & 1
        fill_bit  = 1 - round_bit
        fill_mask = (1 << (shift_amt - 1)) - 1
        rounding_addend = (round_bit << shift_amt) | (fill_bit * fill_mask)

        rounded = product + rounding_addend
        o_sample = (rounded >> shift_amt).astype(np.int32)

        lo = -(1 << (_OW - 1))
        hi = (1 << (_OW - 1)) - 1
        return np.clip(o_sample, lo, hi).astype(np.int32)

    def _fft_frame(self, windowed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Float64 FFT approximation of RTL fftmain (16-bit fixed-point, sign-extended to 18).
        FFT_SCALE=128 compensates for 7 internal ÷2 stages in fftmain."""
        X = np.fft.rfft(windowed.astype(np.float64))

        FFT_SCALE = 128
        re_f = np.real(X) / FFT_SCALE
        im_f = np.imag(X) / FFT_SCALE

        limit = (1 << (FFT_W - 1)) - 1
        re_q = np.clip(np.round(re_f), -limit - 1, limit).astype(np.int64)
        im_q = np.clip(np.round(im_f), -limit - 1, limit).astype(np.int64)

        re_u = re_q & ((1 << FFT_W) - 1)
        im_u = im_q & ((1 << FFT_W) - 1)
        return re_u.astype(np.uint64), im_u.astype(np.uint64)

    def _power(self, re: np.ndarray, im: np.ndarray) -> np.ndarray:
        """power_calc.sv: (re² + im²) >> SHIFT"""
        half = 1 << (FFT_W - 1)
        r = re.astype(np.int64)
        i = im.astype(np.int64)
        r = np.where(r >= half, r - (1 << FFT_W), r)
        i = np.where(i >= half, i - (1 << FFT_W), i)
        s = (r * r + i * i) >> SHIFT
        return (s & POWER_MASK).astype(np.uint64)

    def _filterbank(self, power: np.ndarray) -> np.ndarray:
        """mel_filterbank.sv: 54-bit accumulator, no saturation."""
        p = power.astype(np.int64)
        accum = p @ self.fb_dense
        return (accum & ACCUM_MASK).astype(np.uint64)

    def _log_one(self, energy: int) -> int:
        """log_lut.sv: (floor(log2(e)) << Q_FRAC) + lut[frac]. Returns 0 for energy==0."""
        if energy == 0:
            return 0
        lg = int(energy).bit_length() - 1
        max_lg = (1 << (LOG_OUT_W - Q_FRAC)) - 1  # 15
        if lg > max_lg:
            return LOG_MASK  # saturate to 0xFFFF
        mask = (1 << LUT_FRAC) - 1
        if lg >= LUT_FRAC:
            addr = (energy >> (lg - LUT_FRAC)) & mask
        else:
            addr = (energy << (LUT_FRAC - lg)) & mask
        return ((lg << Q_FRAC) + self.log_lut[addr]) & LOG_MASK

    def _log_compress(self, mel_energy: np.ndarray) -> np.ndarray:
        return np.array(
            [self._log_one(int(mel_energy[k])) for k in range(N_MELS)],
            dtype=np.uint16,
        )

    def _process_frame(self, frame_samples: np.ndarray) -> np.ndarray:
        """Run one frame through window → FFT → power → mel → log. Returns (N_MELS,) uint16."""
        windowed = self._window_frame(frame_samples)
        re, im   = self._fft_frame(windowed)
        power    = self._power(re, im)
        mel_e    = self._filterbank(power)
        return self._log_compress(mel_e)

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """Extract log-mel features (bit-accurate to RTL).
        Input: PCM audio (clipped to 14-bit). Output: (N_MELS, n_frames) uint16 Q4.12."""
        samples = np.clip(np.asarray(audio, dtype=np.int32), -SAMPLE_MAX - 1, SAMPLE_MAX)
        n_samples = len(samples)
        n_frames  = max(0, (n_samples - N_FFT) // HOP_LENGTH + 1)

        if n_frames == 0:
            return np.zeros((N_MELS, 0), dtype=np.uint16)

        out = np.zeros((N_MELS, n_frames), dtype=np.uint16)
        for f in range(n_frames):
            start = f * HOP_LENGTH
            frame = samples[start : start + N_FFT]
            out[:, f] = self._process_frame(frame)
        return out

    def extract_float(self, audio: np.ndarray) -> np.ndarray:
        """Same as extract() but returns float32 in log₂ units."""
        return self.extract(audio).astype(np.float32) / (1 << Q_FRAC)

    def get_config(self) -> dict:
        return {
            "sample_rate": SAMPLE_RATE,
            "n_fft": N_FFT,
            "n_mels": N_MELS,
            "hop_length": HOP_LENGTH,
            "window_length": WIN_LENGTH,
            "sample_w": SAMPLE_W,
            "fft_w": FFT_W,
            "shift": SHIFT,
            "weight_w": WEIGHT_W,
            "accum_w": ACCUM_W,
            "log_out_w": LOG_OUT_W,
            "q_frac": Q_FRAC,
        }


if __name__ == "__main__":
    import time

    ext = GoldenExtractor()
    print("GoldenExtractor config:", ext.get_config())

    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    chirp = np.sin(2 * np.pi * (200 * t + (7000 - 200) / 2 * t ** 2))
    audio_i16 = (chirp * SAMPLE_MAX).astype(np.int16)

    t0 = time.perf_counter()
    feats = ext.extract(audio_i16)
    dt = time.perf_counter() - t0

    print(f"Input: {len(audio_i16)} samples ({len(audio_i16)/SAMPLE_RATE:.2f}s)")
    print(f"Output: {feats.shape} (n_mels={feats.shape[0]}, n_frames={feats.shape[1]})")
    print(f"Time: {dt*1000:.1f} ms, dtype: {feats.dtype}")
    print(f"Range: [{feats.min()}, {feats.max()}]  "
          f"(Q4.12: [{feats.min()/4096:.3f}, {feats.max()/4096:.3f}] log₂)")

    feats_f = ext.extract_float(audio_i16)
    print(f"Float range: [{feats_f.min():.4f}, {feats_f.max():.4f}] log₂")
    print(f"Non-zero bins: {(feats_f > 0).sum()} / {feats_f.size}")
