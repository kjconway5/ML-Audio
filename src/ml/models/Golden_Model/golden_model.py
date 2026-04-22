"""
golden_model.py — Bit-accurate Python replica of the RTL feature pipeline.

Replicates the exact fixed-point arithmetic of:
    stfft.sv  →  power_calc.sv  →  mel_filterbank.sv  →  log_lut.sv

Use this as a drop-in replacement for LogMelExtractor when you need training
features that are numerically identical to what the hardware produces.

Usage:
    from golden_model import GoldenExtractor

    extractor = GoldenExtractor()            # loads config + hex data
    features  = extractor.extract(audio_i16) # (n_mels, n_frames) uint16 Q4.12
    features_float = features / 4096.0       # convert to float log₂ units

The output dtype is np.uint16 (Q4.12) — the same format the CNN sees from RTL.
Call .extract_float() for a float32 version suitable for PyTorch training.
"""

import re as _re

import numpy as np
import torch
import os
from pathlib import Path

# ── RTL parameters (must match features_top / logmel_top defaults) ──────────
SAMPLE_RATE = 16000
N_FFT       = 256
N_BINS      = N_FFT // 2 + 1   # 129
N_MELS      = 40
WIN_LENGTH  = 256
HOP_LENGTH  = N_FFT // 2       # 128  (RTL 50% overlap)

SAMPLE_W    = 14                # stfft input width
FFT_W       = 18                # stfft output width
SHIFT       = 6                 # power_calc right-shift
POWER_W     = 2 * FFT_W - SHIFT + 1  # 31
WEIGHT_W    = 16                # mel coefficient width (Q0.15)
FRAC_BITS   = 15                # mel coeff fractional bits
ACCUM_W     = 54                # mel accumulator width
MAX_COEFFS  = 16                # sparse ROM depth per filter
LOG_OUT_W   = 16                # log output width
LUT_FRAC    = 6                 # log LUT address bits
Q_FRAC      = 12                # log output fractional bits

# Masks
POWER_MASK = (1 << POWER_W) - 1
ACCUM_MASK = (1 << ACCUM_W) - 1
LOG_MASK   = (1 << LOG_OUT_W) - 1
SAMPLE_MAX = (1 << (SAMPLE_W - 1)) - 1   # 8191

# Window coefficient scale (matches hanning.hex)
WIN_COEFF_SCALE = 2047

# windowfn.v rounding parameters
_IW = SAMPLE_W
_TW = SAMPLE_W      # TW=IW in stfft.sv
_OW = SAMPLE_W      # OW=IW in stfft.sv
_AW = _IW + _TW     # accumulator width = 28

# ── paths to hex data files ─────────────────────────────────────────────────
_HERE       = Path(__file__).resolve().parent
_DATA_DIR   = _HERE                        # hex files are co-located
_WIN_HEX    = _HERE / "hanning.hex"
_FFT_DIR    = _HERE                        # cmem_*.hex are co-located


def _load_hex(path: Path) -> list[int]:
    """Read a newline-delimited hex file into a list of ints."""
    with open(path) as f:
        return [int(line.strip(), 16) for line in f if line.strip()]


def _load_cmem(path: Path) -> list[tuple[int, int]]:
    """Load ZipCPU twiddle factor hex file → list of (w_r, w_i) 20-bit signed."""
    with open(path) as f:
        raw = f.read()
    result = []
    for tok in _re.findall(r'^[0-9a-fA-F]+', raw, _re.MULTILINE):
        v = int(tok, 16)
        w_r = (v >> 20) & 0xFFFFF
        w_i = v & 0xFFFFF
        if w_r >> 19: w_r -= (1 << 20)
        if w_i >> 19: w_i -= (1 << 20)
        result.append((w_r, w_i))
    return result


def _cr_half(val: int) -> int:
    """convround(17→16, SHIFT=0) DROP_ONE_BIT: banker's round(val / 2)."""
    trunc = val >> 1          # arithmetic right shift
    if (val & 1) and (trunc & 1):
        trunc += 1
    result = trunc & 0xFFFF
    if result >= 0x8000:
        result -= 0x10000
    return result


def _cr_mpy(val: int) -> int:
    """convround(39→16, SHIFT=4) ROUND_RESULT: banker's round of butterfly product."""
    trunc = val >> 19
    first_lost = (val >> 18) & 1
    other_lost = val & ((1 << 18) - 1)
    if first_lost and (other_lost or (trunc & 1)):
        trunc += 1
    result = trunc & 0xFFFF
    if result >= 0x8000:
        result -= 0x10000
    return result


def _cr_last(val: int) -> int:
    """convround(17→16, SHIFT=1) SHIFT_ONE_BIT: lower 16 bits."""
    result = val & 0xFFFF
    if result >= 0x8000:
        result -= 0x10000
    return result


def _bit_rev8(x: int) -> int:
    """Reverse 8-bit integer for bitreverse stage."""
    x = ((x & 0xF0) >> 4) | ((x & 0x0F) << 4)
    x = ((x & 0xCC) >> 2) | ((x & 0x33) << 2)
    x = ((x & 0xAA) >> 1) | ((x & 0x55) << 1)
    return x


class GoldenExtractor:
    """Bit-accurate replica of the RTL STFFT → Log-Mel pipeline."""

    def __init__(self):
        # Window coefficients (periodic Hann, 12-bit scale)
        self.win_coeffs = np.array(_load_hex(_WIN_HEX), dtype=np.int32)
        assert len(self.win_coeffs) == N_FFT, (
            f"hanning.hex has {len(self.win_coeffs)} entries, expected {N_FFT}"
        )

        # Mel filterbank (Q0.15, sparse ROM layout)
        raw_coeffs = _load_hex(_DATA_DIR / "mel_coeffs.hex")
        self.mel_starts = _load_hex(_DATA_DIR / "mel_starts.hex")
        self.mel_ends   = _load_hex(_DATA_DIR / "mel_ends.hex")
        max_coeffs_actual = len(raw_coeffs) // N_MELS
        self.mel_coeffs = np.array(raw_coeffs, dtype=np.int64).reshape(N_MELS, max_coeffs_actual)

        # Build a dense (N_BINS, N_MELS) filterbank for matrix multiply.
        # This is equivalent to the sparse ROM iteration in mel_filterbank.sv.
        self.fb_dense = np.zeros((N_BINS, N_MELS), dtype=np.int64)
        for m in range(N_MELS):
            s = self.mel_starts[m]
            e = self.mel_ends[m]
            n_c = e - s + 1
            self.fb_dense[s:e+1, m] = self.mel_coeffs[m, :n_c]

        # Log2 LUT (Q4.12)
        self.log_lut = _load_hex(_DATA_DIR / "log2_lut.hex")
        assert len(self.log_lut) == (1 << LUT_FRAC)

        # FFT twiddle factors (ZipCPU cmem_*.hex, one per fftstage)
        self._fft_cmems = {
            lgspan: _load_cmem(_FFT_DIR / f"cmem_{1 << (lgspan + 1)}.hex")
            for lgspan in range(2, 8)
        }

    # ── Stage 1: Windowing ──────────────────────────────────────────────────
    def _window_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Replicate windowfn.v: signed multiply + convergent rounding.

        windowfn.v BIT_ADJUSTMENT_ROUNDING (OW < AW):
            rounded = product + { {OW{0}}, product[AW-OW], {(AW-OW-1){!product[AW-OW]}} }
            o_sample = rounded[AW-1 : AW-OW]

        With IW=OW=TW=14, AW=28:
            shift_amt = AW - OW = 14
            round_bit = product[14]
            fill_bit  = ~product[14]   (13 copies)
            rounding_addend = (round_bit << 14) | (fill_bit * 0x1FFF)
                            = product[14] << 14  |  (~product[14] & 1) * 0x1FFF
            rounded = product + rounding_addend
            o_sample = rounded >> 14   (upper OW bits)
        """
        # frame is SAMPLE_W-bit signed integers
        data = frame.astype(np.int64)
        tap  = self.win_coeffs.astype(np.int64)

        product = data * tap  # signed (AW-bit)

        shift_amt = _AW - _OW  # 14
        round_bit = (product >> shift_amt) & 1
        fill_bit  = 1 - round_bit  # ~round_bit (1-bit)
        fill_mask = (1 << (shift_amt - 1)) - 1  # 0x1FFF for 13 bits
        rounding_addend = (round_bit << shift_amt) | (fill_bit * fill_mask)

        rounded = product + rounding_addend
        o_sample = (rounded >> shift_amt).astype(np.int32)

        # Clamp to OW-bit signed range (shouldn't be needed but mirrors HW)
        lo = -(1 << (_OW - 1))
        hi = (1 << (_OW - 1)) - 1
        return np.clip(o_sample, lo, hi).astype(np.int32)

    # ── Stage 2: FFT ────────────────────────────────────────────────────────
    def _fft_frame(self, windowed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Bit-accurate 256-point DIF FFT replicating fftmain.v.

        Implements the ZipCPU pipelined FFT (./fftgen -f 256 -n 16 -m 16 -k 4):
          6× fftstage  (IWIDTH=16, CWIDTH=20, OWIDTH=16, BFLYSHIFT=0, CKPCE=3)
          1× qtrstage  (IWIDTH=16, OWIDTH=16, SHIFT=0)
          1× laststage (IWIDTH=16, OWIDTH=16, SHIFT=1)
          1× bitreverse (LGSIZE=8)

        Butterfly rounding per convround.v (convergent / banker's round):
          Left output  : _cr_half  — convround(39→16, SHIFT=4) on sum<<18
          Right output : _cr_mpy   — convround(39→16, SHIFT=4) on product
          qtrstage     : _cr_half  — convround(17→16, SHIFT=0) DROP_ONE_BIT
          laststage    : _cr_last  — convround(17→16, SHIFT=1) SHIFT_ONE_BIT

        Twiddle factors loaded from cmem_*.hex (real upper 20 bits,
        imag lower 20 bits, scaled by 2^18).
        """
        N = N_FFT  # 256
        dr = [int(x) for x in windowed]  # 16-bit signed (sign-ext from 14-bit)
        di = [0] * N

        # ── 6 fftstages ───────────────────────────────────────────────────
        for lgspan in (7, 6, 5, 4, 3, 2):
            span   = 1 << lgspan
            groups = N // (2 * span)
            tw     = self._fft_cmems[lgspan]
            for k in range(span):
                w_r, w_i = tw[k]
                for g in range(groups):
                    i = g * 2 * span + k
                    j = i + span
                    lr, li = dr[i], di[i]
                    rr, ri = dr[j], di[j]
                    sum_r, sum_i = lr + rr, li + ri
                    dif_r, dif_i = lr - rr, li - ri
                    dr[i] = _cr_half(sum_r)
                    di[i] = _cr_half(sum_i)
                    dr[j] = _cr_mpy(dif_r * w_r - dif_i * w_i)
                    di[j] = _cr_mpy(dif_r * w_i + dif_i * w_r)

        # ── qtrstage (4-point DIF, IWIDTH=16, OWIDTH=16, SHIFT=0) ─────────
        # Pairs: (4g+k, 4g+k+2) for g=0..63, k=0..1
        # k=0: twiddle W=1; k=1: twiddle W=-j (forward)
        for g in range(64):
            for k in range(2):
                i = g * 4 + k
                j = i + 2
                lr, li = dr[i], di[i]
                rr, ri = dr[j], di[j]
                sum_r, sum_i = lr + rr, li + ri
                dif_r, dif_i = lr - rr, li - ri
                dr[i] = _cr_half(sum_r)
                di[i] = _cr_half(sum_i)
                rdr = _cr_half(dif_r)
                rdi = _cr_half(dif_i)
                if k == 0:                    # W = 1
                    dr[j], di[j] = rdr, rdi
                else:                          # W = -j: (a+jb)(-j) = b - ja
                    dr[j], di[j] = rdi, -rdr

        # ── laststage (2-point DIF, IWIDTH=16, OWIDTH=16, SHIFT=1) ────────
        # Pairs: (2k, 2k+1) for k=0..127
        for k in range(128):
            i = 2 * k
            j = i + 1
            lr, li = dr[i], di[i]
            rr, ri = dr[j], di[j]
            dr[i] = _cr_last(lr + rr)
            di[i] = _cr_last(li + ri)
            dr[j] = _cr_last(lr - rr)
            di[j] = _cr_last(li - ri)

        # ── bitreverse (LGSIZE=8) ──────────────────────────────────────────
        out_r = [0] * N
        out_i = [0] * N
        for k in range(N):
            out_r[_bit_rev8(k)] = dr[k]
            out_i[_bit_rev8(k)] = di[k]

        # First N_BINS=129 bins; sign-extend 16→18 bits (same signed value)
        re_q = np.array(out_r[:N_BINS], dtype=np.int64)
        im_q = np.array(out_i[:N_BINS], dtype=np.int64)
        re_u = re_q & ((1 << FFT_W) - 1)
        im_u = im_q & ((1 << FFT_W) - 1)
        return re_u.astype(np.uint64), im_u.astype(np.uint64)

    # ── Stage 3: Power spectrum ─────────────────────────────────────────────
    def _power(self, re: np.ndarray, im: np.ndarray) -> np.ndarray:
        """
        power_calc.sv:  power = (re² + im²) >> SHIFT

        Inputs are FFT_W-bit unsigned (2's complement representation).
        Sign-extend to signed, square, sum, shift.
        """
        half = 1 << (FFT_W - 1)
        r = re.astype(np.int64)
        i = im.astype(np.int64)
        r = np.where(r >= half, r - (1 << FFT_W), r)
        i = np.where(i >= half, i - (1 << FFT_W), i)
        s = (r * r + i * i) >> SHIFT
        return (s & POWER_MASK).astype(np.uint64)

    # ── Stage 4: Mel filterbank ─────────────────────────────────────────────
    def _filterbank(self, power: np.ndarray) -> np.ndarray:
        """
        mel_filterbank.sv:  accum += power_buf[bin] * weight

        Uses the dense filterbank matrix (numerically identical to sparse ROM).
        54-bit unsigned accumulator, no saturation.
        """
        p = power.astype(np.int64)
        accum = p @ self.fb_dense  # (N_BINS,) @ (N_BINS, N_MELS) → (N_MELS,)
        return (accum & ACCUM_MASK).astype(np.uint64)

    # ── Stage 5: Log₂ compression ──────────────────────────────────────────
    def _log_one(self, energy: int) -> int:
        """
        log_lut.sv: result = (floor(log2(e)) << Q_FRAC) + lut[frac_addr]

        Returns 0 for energy == 0.
        Saturates at LOG_MASK (0xFFFF) when log2_int exceeds Q4.12 range.
        """
        if energy == 0:
            return 0
        lg   = int(energy).bit_length() - 1
        max_lg = (1 << (LOG_OUT_W - Q_FRAC)) - 1   # 15
        if lg > max_lg:
            return LOG_MASK   # saturate to 0xFFFF
        mask = (1 << LUT_FRAC) - 1
        if lg >= LUT_FRAC:
            addr = (energy >> (lg - LUT_FRAC)) & mask
        else:
            addr = (energy << (LUT_FRAC - lg)) & mask
        return ((lg << Q_FRAC) + self.log_lut[addr]) & LOG_MASK

    def _log_compress(self, mel_energy: np.ndarray) -> np.ndarray:
        """Apply log₂ LUT to all N_MELS energies."""
        return np.array(
            [self._log_one(int(mel_energy[k])) for k in range(N_MELS)],
            dtype=np.uint16,
        )

    # ── Full frame pipeline ─────────────────────────────────────────────────
    def _process_frame(self, frame_samples: np.ndarray) -> np.ndarray:
        """
        Process one FFT frame through the full pipeline.
        frame_samples: (N_FFT,) array of SAMPLE_W-bit signed integers.
        Returns: (N_MELS,) uint16 Q4.12 log-mel features.
        """
        windowed = self._window_frame(frame_samples)
        re, im   = self._fft_frame(windowed)
        power    = self._power(re, im)
        mel_e    = self._filterbank(power)
        log_mel  = self._log_compress(mel_e)
        return log_mel

    # ── Public API ──────────────────────────────────────────────────────────
    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract log-mel features from raw audio (bit-accurate to RTL).

        Args:
            audio: int16 (or int32) PCM audio at SAMPLE_RATE.
                   Values are clipped to SAMPLE_W-bit signed range.

        Returns:
            features: np.uint16 array, shape (N_MELS, n_frames), Q4.12.
                      Each column is one time frame.

        Framing matches RTL: hop = N_FFT // 2 = 128, no center padding.
        """
        # Clip to SAMPLE_W-bit signed range (RTL input is 14-bit)
        samples = np.clip(
            np.asarray(audio, dtype=np.int32),
            -SAMPLE_MAX - 1,
            SAMPLE_MAX,
        )

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
        """
        Same as extract() but returns float32 in log₂ units.

        Suitable for direct use as PyTorch training features:
            features = extractor.extract_float(audio_i16)
            tensor   = torch.from_numpy(features)  # (N_MELS, n_frames)
        """
        q12 = self.extract(audio)
        return q12.astype(np.float32) / (1 << Q_FRAC)

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


# ── Convenience: standalone test ────────────────────────────────────────────
if __name__ == "__main__":
    import time

    ext = GoldenExtractor()
    print("GoldenExtractor config:", ext.get_config())

    # Quick smoke test: 1-second chirp
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    chirp = np.sin(2 * np.pi * (200 * t + (7000 - 200) / 2 * t ** 2))
    audio_i16 = (chirp * SAMPLE_MAX).astype(np.int16)

    t0 = time.perf_counter()
    feats = ext.extract(audio_i16)
    dt = time.perf_counter() - t0

    print(f"Input: {len(audio_i16)} samples ({len(audio_i16)/SAMPLE_RATE:.2f}s)")
    print(f"Output: {feats.shape} (n_mels={feats.shape[0]}, n_frames={feats.shape[1]})")
    print(f"Time: {dt*1000:.1f} ms")
    print(f"dtype: {feats.dtype}")
    print(f"Range: [{feats.min()}, {feats.max()}]  "
          f"(Q4.12: [{feats.min()/4096:.3f}, {feats.max()/4096:.3f}] log₂)")

    # Also show float version
    feats_f = ext.extract_float(audio_i16)
    print(f"\nFloat output range: [{feats_f.min():.4f}, {feats_f.max():.4f}] log₂")
    print(f"Non-zero bins: {(feats_f > 0).sum()} / {feats_f.size}")
