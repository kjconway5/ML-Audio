"""
golden_model.py
===============
Bit-accurate golden model for the RTL audio feature pipeline.

Contains two extractors:

  GoldenExtractor
    -- STFFT-only pipeline (pipeline_top.sv, 16 kHz PCM input)
    -- Stages: Hann window -> R2FFT (BFP) -> power -> mel filterbank
                -> log2 LUT -> bfpexp compensation

  FullPipelineGoldenExtractor
    -- Full pipeline (full_pipeline_top.sv, PDM microphone input)
    -- Stages: PCM -> sigma-delta PDM -> CIC decimator -> truncation
                -> compFIR -> truncation -> all STFFT stages above

Both classes produce (N_MELS, n_frames) uint16 in Q_FRAC fixed-point log2.
Call .extract_float() for float32 log2 values.
"""

import numpy as np
import torch
import os
from pathlib import Path

# RTL parameters  (must match logmel_top defaults)
SAMPLE_RATE = 16000
N_FFT       = 256
N_BINS      = N_FFT // 2 + 1   # 129
N_MELS      = 40
WIN_LENGTH  = 256
HOP_LENGTH  = N_FFT // 2       # 128

SAMPLE_W    = 16
FFT_W       = 16
SHIFT       = 6
POWER_W     = 2 * FFT_W - SHIFT + 1   # 27
WEIGHT_W    = 16
FRAC_BITS   = 15
ACCUM_W     = 54
MAX_COEFFS  = 16
LOG_OUT_W   = 16
LUT_FRAC    = 6
Q_FRAC      = 10

POWER_MASK = (1 << POWER_W) - 1
ACCUM_MASK = (1 << ACCUM_W) - 1
LOG_MASK   = (1 << LOG_OUT_W) - 1
SAMPLE_MAX = (1 << (SAMPLE_W - 1)) - 1   # 32767

WIN_COEFF_SCALE = 2047

# windowfn.v rounding parameters
_IW = SAMPLE_W
_TW = SAMPLE_W
_OW = SAMPLE_W
_AW = _IW + _TW   # 32

# Hex data paths
_HERE      = Path(__file__).resolve().parent
_DATA_DIR  = _HERE.parent / "rtl" / "Log-Mel" / "data"
_WIN_HEX   = _HERE.parent / "rtl" / "STFFT" / "ZipCPU" / "hanning.hex"


def _load_hex(path: Path) -> list:
    with open(path) as f:
        return [int(line.strip(), 16) for line in f if line.strip()]



# R2FFT emulation
# Bit-accurate replica of the RTL's per-stage BFP + 16-bit radix-2 butterfly.
# Matches:
#   radix2Butterfly.sv       (butterfly arithmetic)
#   bfp_Shifter.sv           (operand normalisation)
#   bfp_bitWidthAcc.sv       (bfpexp accumulation)
#   bfp_bitWidthDetector.sv  (stage bit-width detection)

def _bit_reverse_perm(x: np.ndarray, log2N: int) -> np.ndarray:
    N   = x.shape[0]
    idx = np.arange(N)
    rev = np.zeros(N, dtype=np.int64)
    for b in range(log2N):
        rev |= ((idx >> b) & 1) << (log2N - 1 - b)
    return x[rev]


def _bit_width(v: int, DW: int = 16) -> int:
    v = abs(int(v))
    if v == 0:
        return 0
    return min(v.bit_length(), DW)


def _bfp_scale(bw: int, DW: int = 16) -> int:
    if bw == DW: return 1
    if bw == 0:  return 0
    return bw - (DW - 2)


def _rtl_butterfly(a_re, a_im, b_re, b_im, w_re, w_im, DW=16):
    """Exact radix2Butterfly.sv arithmetic."""
    a_re, a_im = int(a_re), int(a_im)
    b_re, b_im = int(b_re), int(b_im)
    w_re, w_im = int(w_re), int(w_im)

    dst_a_re = (a_re + b_re) >> 1
    dst_a_im = (a_im + b_im) >> 1
    xbuf_re  = (a_re - b_re) >> 1
    xbuf_im  = (a_im - b_im) >> 1

    xbuf_re_p_im = xbuf_re + xbuf_im
    tw_re_p_im   = w_re + w_im
    tw_re_m_im   = w_re - w_im

    tmp_a = xbuf_re_p_im * w_re
    tmp_r = tw_re_p_im   * xbuf_im
    tmp_i = tw_re_m_im   * xbuf_re

    round_add = 1 << (DW - 2)
    yr = (tmp_a - tmp_r + round_add) >> (DW - 1)
    yi = (tmp_a - tmp_i + round_add) >> (DW - 1)

    hi = (1 << (DW - 1)) - 1
    lo = -(1 << (DW - 1))
    return dst_a_re, dst_a_im, max(lo, min(hi, yr)), max(lo, min(hi, yi))


def _r2fft_emulate(x_real: np.ndarray, FFT_W: int = 16):
    """Emulate R2FFT. Returns (re_u, im_u, bfpexp)."""
    DW    = FFT_W
    N     = x_real.shape[0]
    log2N = int(np.log2(N))
    assert 1 << log2N == N

    hi = (1 << (DW - 1)) - 1
    lo = -(1 << (DW - 1))
    X_re = [int(max(lo, min(hi, int(v)))) for v in x_real]
    X_im = [0] * N

    max_abs_in = max((abs(v) for v in X_re), default=0)
    bw     = _bit_width(max_abs_in, DW)
    bfpexp = _bfp_scale(bw, DW)

    tw_lim = 1 << (DW - 1)
    tw_re  = [0] * N
    tw_im  = [0] * N
    for k in range(N):
        ang    = -2.0 * np.pi * k / N
        tw_re[k] = max(-tw_lim, min(tw_lim, int(round(np.cos(ang) * tw_lim))))
        tw_im[k] = max(-tw_lim, min(tw_lim, int(round(np.sin(ang) * tw_lim))))

    group_size = N
    for s in range(log2N):
        half = group_size >> 1

        if bw == 0 or bw == DW or bw == DW - 1:
            Xs_re, Xs_im = X_re, X_im
        else:
            shift = (DW - 1) - bw
            Xs_re = [v << shift for v in X_re]
            Xs_im = [v << shift for v in X_im]

        new_re  = [0] * N
        new_im  = [0] * N
        max_out = 0
        tw_stride = 1 << s

        for g in range(0, N, group_size):
            for k in range(half):
                ai = g + k
                bi = g + k + half
                wr = tw_re[k * tw_stride]
                wi = tw_im[k * tw_stride]
                ya_re, ya_im, yb_re, yb_im = _rtl_butterfly(
                    Xs_re[ai], Xs_im[ai], Xs_re[bi], Xs_im[bi], wr, wi, DW)
                new_re[ai] = ya_re; new_im[ai] = ya_im
                new_re[bi] = yb_re; new_im[bi] = yb_im
                for v in (ya_re, ya_im, yb_re, yb_im):
                    av = abs(v)
                    if av > max_out: max_out = av

        X_re, X_im = new_re, new_im
        bw = _bit_width(max_out, DW)
        if s < log2N - 1:
            bfpexp += _bfp_scale(bw, DW)
        group_size = half

    Xr_nat = _bit_reverse_perm(np.asarray(X_re, dtype=np.int64), log2N)
    Xi_nat = _bit_reverse_perm(np.asarray(X_im, dtype=np.int64), log2N)
    nbins  = N // 2 + 1
    mask   = (1 << DW) - 1
    return ((Xr_nat[:nbins] & mask).astype(np.uint64),
            (Xi_nat[:nbins] & mask).astype(np.uint64),
            int(bfpexp))


# CIC Decimator emulation
#
# Critical: all always @(posedge clk) blocks are simultaneous.
# When cycle==0, the comb stages fire using OLD int_reg values -- BEFORE
# the new sample is added to the integrators this cycle.
# Integrators also update simultaneously using OLD values of each other.

def _s(v: int, bits: int) -> int:
    """Wrap to signed bits-wide two's complement."""
    mask = (1 << bits) - 1
    v    = int(v) & mask
    return v - (1 << bits) if v >= (1 << (bits - 1)) else v


class _CICDecimator:
    """
    N=3, M=1, R=63 CIC decimator matching cic_decimator.v.

    All registers are signed 34-bit (REG_WIDTH = 16 + ceil(log2(63^3)) = 34).
    Integrators run every sample; combs fire once per R samples.
    Simultaneous register update semantics (Verilog non-blocking assignments).
    """
    N   = 3
    M   = 1
    R   = 63
    REG = 34    # bits

    def __init__(self):
        self._reset()

    def _reset(self):
        self._int   = [0] * self.N
        self._comb  = [0] * self.N
        self._delay = [0] * self.N   # M=1: one delay per comb stage
        self._cycle = 0

    def process(self, samples: np.ndarray) -> np.ndarray:
        """
        Feed int16/int32 samples (PDM values: +32767 or -32768).
        Returns int64 array of 34-bit signed CIC outputs, one per R samples.
        """
        self._reset()
        B   = self.REG
        out = []

        for x in samples:
            x = int(x)

            # Step 1: comb stages fire FIRST using OLD integrator values
            # (matches Verilog: comb block reads int_reg before integrator block updates it)
            if self._cycle == 0:
                i2_old = self._int[2]
                c0_old = self._comb[0]
                c1_old = self._comb[1]
                d0, d1, d2 = self._delay

                new_c0 = _s(i2_old - d0, B)
                new_c1 = _s(c0_old - d1, B)
                new_c2 = _s(c1_old - d2, B)

                self._comb  = [new_c0, new_c1, new_c2]
                self._delay = [i2_old, c0_old, c1_old]
                out.append(new_c2)

            # Step 2: integrators update simultaneously using OLD values of each other
            i0_old, i1_old, i2_old = self._int
            self._int[0] = _s(i0_old + x,      B)
            self._int[1] = _s(i1_old + i0_old, B)
            self._int[2] = _s(i2_old + i1_old, B)

            # Step 3: advance cycle counter
            self._cycle = 0 if self._cycle >= self.R - 1 else self._cycle + 1

        return np.array(out, dtype=np.int64)

    @staticmethod
    def truncate(raw: np.ndarray) -> np.ndarray:
        """
        Keep bits [33:18] of the 34-bit accumulator.
        Matches: assign cic_trunc = cic_data[CIC_REG_W-1 : CIC_REG_W-16];
        """
        shifted = raw >> 18
        masked  = shifted & 0xFFFF
        return np.where(masked >= (1 << 15),
                        masked - (1 << 16), masked).astype(np.int16)


# compFIR emulation  (matches compFIR.sv with sr_next delay line)
#
# The sr_next structure means:
#   output[n] uses [x[n], x[n-1], ..., x[n-32]] as the delay line.
# Equivalent to a standard causal FIR with the current sample already included
# (1-cycle latency).  Python implementation: shift first, then convolve.

class _CompFIR:
    """
    33-tap CIC compensation FIR matching compFIR.sv.
    Half-coefficients (k=0 outermost, k=16 centre) from the Verilog CSD assigns.
    OW = 37 bits (IW=16 + CW=14 + ceil(log2(33)) + 1).
    """
    _HALF = np.array([
        11952, -2084, -77, 699, -845, 802, -680, 533, -390, 267,
        -169, 99, -53, 25, -10, 3, -1
    ], dtype=np.int64)

    NTAPS = 33
    OW    = 37

    def __init__(self):
        M    = (self.NTAPS - 1) // 2
        h    = np.zeros(self.NTAPS, dtype=np.int64)
        for k in range(M):
            h[k]              = self._HALF[k]
            h[self.NTAPS-1-k] = self._HALF[k]
        h[M] = self._HALF[M]
        self._h   = h
        self._max =  (1 << (self.OW - 1)) - 1
        self._min = -(1 << (self.OW - 1))

    def process(self, samples: np.ndarray) -> np.ndarray:
        """
        Process int16 array through the FIR.
        Returns int64 array of OW=37-bit full-precision outputs.
        """
        buf = np.zeros(self.NTAPS, dtype=np.int64)
        out = np.empty(len(samples), dtype=np.int64)
        for i, x in enumerate(samples):
            buf[1:] = buf[:-1]
            buf[0]  = int(x)
            y       = int(np.dot(buf, self._h))
            out[i]  = max(self._min, min(self._max, y))
        return out

    @staticmethod
    def truncate(raw: np.ndarray) -> np.ndarray:
        """
        Keep bits [30:15] of the 37-bit output.
        Matches: assign fir_trunc = fir_tdata[30:15];
        """
        shifted = raw >> 15
        masked  = shifted & 0xFFFF
        return np.where(masked >= (1 << 15),
                        masked - (1 << 16), masked).astype(np.int16)


# Sigma-delta PDM modulator
# Matches pcm_to_pdm() + drive_pdm() in the cocotb testbench exactly.

def _pcm_to_cic_input(pcm: np.ndarray, decim: int = 63) -> np.ndarray:
    """
    First-order sigma-delta modulator.
    Each PCM sample -> decim PDM bits.
    PDM bit 1 -> +32767, PDM bit 0 -> -32768  (matches drive_pdm in testbench).
    Returns int16 array of length len(pcm)*decim fed into the CIC.
    """
    vals = []
    acc  = 0
    for x in pcm:
        for _ in range(decim):
            acc += int(x)
            if acc >= 0:
                vals.append(0x7FFF)
                acc -= (1 << 15)
            else:
                vals.append(-0x8000)
                acc += (1 << 15)
    return np.array(vals, dtype=np.int16)


# GoldenExtractor  (STFFT-only pipeline -- pipeline_top.sv)

class GoldenExtractor:
    """
    Bit-accurate replica of the STFFT-only RTL pipeline (pipeline_top.sv).

    Stages:
      Hann window -> R2FFT (per-stage BFP) -> power (>>SHIFT) ->
      sparse mel filterbank -> log2 LUT -> bfpexp compensation

    Input:  16-bit signed PCM at 16 kHz
    Output: (N_MELS, n_frames) uint16 in Q_FRAC fixed-point log2 units

    Set bfp_compensate=False to get raw pre-compensation values
    (matches u_logmel.cnn_data_ol before the mel_compensated_o adder).
    """

    def __init__(self, bfp_compensate: bool = True):
        self.bfp_compensate = bfp_compensate

        self.win_coeffs = np.array(_load_hex(_WIN_HEX), dtype=np.int32)
        assert len(self.win_coeffs) == N_FFT

        raw_coeffs      = _load_hex(_DATA_DIR / "mel_coeffs.hex")
        self.mel_starts = _load_hex(_DATA_DIR / "mel_starts.hex")
        self.mel_ends   = _load_hex(_DATA_DIR / "mel_ends.hex")
        self.mel_coeffs = np.array(raw_coeffs, dtype=np.int64).reshape(N_MELS, MAX_COEFFS)

        self.fb_dense = np.zeros((N_BINS, N_MELS), dtype=np.int64)
        for m in range(N_MELS):
            s = self.mel_starts[m]
            e = self.mel_ends[m]
            self.fb_dense[s:e+1, m] = self.mel_coeffs[m, :e - s + 1]

        self.log_lut = _load_hex(_DATA_DIR / "log2_lut.hex")
        assert len(self.log_lut) == (1 << LUT_FRAC)

    def _window_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        windowfn.v: a_win_samp_w = a_s2_prod[2*IW-2 : IW-1]
        Arithmetic right-shift by (IW-1)=15, take low IW bits as signed.
        """
        product   = frame.astype(np.int64) * self.win_coeffs.astype(np.int64)
        shifted   = product >> (_IW - 1)
        mask      = (1 << _OW) - 1
        wrapped   = shifted & mask
        sign_bit  = 1 << (_OW - 1)
        return np.where(wrapped >= sign_bit,
                        wrapped - (1 << _OW), wrapped).astype(np.int32)

    def _fft_frame(self, windowed: np.ndarray):
        """R2FFT emulation. Returns (re_u, im_u, bfpexp)."""
        return _r2fft_emulate(windowed.astype(np.int64), FFT_W=FFT_W)

    def _power(self, re: np.ndarray, im: np.ndarray) -> np.ndarray:
        """power_calc.sv: (re^2 + im^2) >> SHIFT, masked to POWER_W bits."""
        half = 1 << (FFT_W - 1)
        r = re.astype(np.int64)
        i = im.astype(np.int64)
        r = np.where(r >= half, r - (1 << FFT_W), r)
        i = np.where(i >= half, i - (1 << FFT_W), i)
        return ((r * r + i * i) >> SHIFT & POWER_MASK).astype(np.uint64)

    def _filterbank(self, power: np.ndarray) -> np.ndarray:
        """mel_filterbank.sv: 54-bit accumulator, no saturation."""
        return (power.astype(np.int64) @ self.fb_dense & ACCUM_MASK).astype(np.uint64)

    def _log_one(self, energy: int) -> int:
        """log_lut.sv: (floor(log2(e)) << Q_FRAC) + lut[frac]."""
        if energy == 0:
            return 0
        lg = int(energy).bit_length() - 1
        if lg > (1 << (LOG_OUT_W - Q_FRAC)) - 1:
            return LOG_MASK
        mask = (1 << LUT_FRAC) - 1
        addr = (energy >> (lg - LUT_FRAC)) & mask if lg >= LUT_FRAC \
               else (energy << (LUT_FRAC - lg)) & mask
        return ((lg << Q_FRAC) + self.log_lut[addr]) & LOG_MASK

    def _log_compress(self, mel_energy: np.ndarray) -> np.ndarray:
        return np.array([self._log_one(int(mel_energy[k])) for k in range(N_MELS)],
                        dtype=np.uint16)

    def _bfp_compensate(self, log_vals: np.ndarray, bfpexp: int) -> np.ndarray:
        """
        pipeline_top.sv bfpexp correction:
          mel_compensated = log_val + 2 * bfpexp * 2^Q_FRAC
        """
        correction = 2 * int(bfpexp) * (1 << Q_FRAC)
        widened    = log_vals.astype(np.int64) + correction
        return np.clip(widened, 0, (1 << LOG_OUT_W) - 1).astype(np.uint16)

    def _process_frame(self, frame_samples: np.ndarray) -> np.ndarray:
        windowed        = self._window_frame(frame_samples)
        re, im, bfpexp  = self._fft_frame(windowed)
        power           = self._power(re, im)
        mel_e           = self._filterbank(power)
        log_pre         = self._log_compress(mel_e)
        if not self.bfp_compensate:
            return log_pre
        return self._bfp_compensate(log_pre, bfpexp)

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract log-mel features.
        Input:  1-D int32/int16 PCM at 16 kHz.
        Output: (N_MELS, n_frames) uint16 Q_FRAC fixed-point log2.
        """
        samples  = np.clip(np.asarray(audio, dtype=np.int32),
                           -SAMPLE_MAX - 1, SAMPLE_MAX)
        n        = len(samples)
        n_frames = max(0, (n - N_FFT) // HOP_LENGTH + 1)
        if n_frames == 0:
            return np.zeros((N_MELS, 0), dtype=np.uint16)
        out = np.zeros((N_MELS, n_frames), dtype=np.uint16)
        for f in range(n_frames):
            s         = f * HOP_LENGTH
            out[:, f] = self._process_frame(samples[s : s + N_FFT])
        return out

    def extract_float(self, audio: np.ndarray) -> np.ndarray:
        """Same as extract() but returns float32 log2 values."""
        return self.extract(audio).astype(np.float32) / (1 << Q_FRAC)

    def get_config(self) -> dict:
        return dict(
            sample_rate=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS,
            hop_length=HOP_LENGTH, window_length=WIN_LENGTH,
            sample_w=SAMPLE_W, fft_w=FFT_W, shift=SHIFT,
            weight_w=WEIGHT_W, accum_w=ACCUM_W,
            log_out_w=LOG_OUT_W, q_frac=Q_FRAC,
        )


# FullPipelineGoldenExtractor  (full pipeline -- full_pipeline_top.sv)

class FullPipelineGoldenExtractor:
    """
    Bit-accurate golden model for the full RTL pipeline (full_pipeline_top.sv).

    Stages:
      PCM -> sigma-delta PDM -> CIC decimator (N=3, M=1, R=63) ->
      truncation [33:18] -> compFIR (33 taps, CSD) -> truncation [30:15] ->
      Hann window -> R2FFT (BFP) -> power -> mel filterbank ->
      log2 LUT -> bfpexp compensation

    The PDM conversion exactly matches the cocotb testbench:
      pcm_to_pdm():  first-order sigma-delta, DECIM=63 bits per sample
      drive_pdm():   bit 1 -> +32767, bit 0 -> -32768

    Input:  1-D int32/int16 PCM at 16 kHz  (same as make_chirp/make_yes/etc.)
    Output: (N_MELS, n_frames) uint16 Q_FRAC fixed-point log2

    Truncation parameters (must match full_pipeline_top.sv):
      CIC:  cic_data[33:18]   -> shift=18
      FIR:  fir_tdata[30:15]  -> shift=15  (sum(full_h)=20143)
    """

    DECIM = 63

    def __init__(self, bfp_compensate: bool = True):
        self._stfft = GoldenExtractor(bfp_compensate=bfp_compensate)
        self._cic   = _CICDecimator()
        self._fir   = _CompFIR()
        self.bfp_compensate = bfp_compensate

    def extract(self, pcm: np.ndarray) -> np.ndarray:
        """
        Full pipeline extraction.

        Args:
            pcm: 1-D int32/int16 PCM at 16 kHz.
                 This is the same array produced by make_chirp(), _load_wav(), etc.

        Returns:
            (N_MELS, n_frames) uint16 matching mel_compensated_o in the RTL.
        """
        pcm = np.clip(np.asarray(pcm, dtype=np.int32), -32768, 32767)

        # Stage 1: PCM -> PDM -> CIC input stream (+/-32767/32768)
        cic_in = _pcm_to_cic_input(pcm, decim=self.DECIM)

        # Stage 2: CIC decimation (34-bit signed, simultaneous updates)
        cic_raw = self._cic.process(cic_in)

        # Stage 3: CIC truncation [33:18] -> 16-bit signed
        cic_trunc = _CICDecimator.truncate(cic_raw)

        # Stage 4: compFIR (OW=37-bit output, sr_next structure)
        fir_raw = self._fir.process(cic_trunc)

        # Stage 5: FIR truncation [30:15] -> 16-bit signed
        fir_trunc = _CompFIR.truncate(fir_raw)

        # Stages 6-9: STFFT + LogMel (reuse GoldenExtractor internals)
        n        = len(fir_trunc)
        n_frames = max(0, (n - N_FFT) // HOP_LENGTH + 1)
        if n_frames == 0:
            return np.zeros((N_MELS, 0), dtype=np.uint16)

        b   = self._stfft
        out = np.zeros((N_MELS, n_frames), dtype=np.uint16)
        for f in range(n_frames):
            s        = f * HOP_LENGTH
            frame    = fir_trunc[s : s + N_FFT].astype(np.int32)
            windowed = b._window_frame(frame)
            re, im, bfpexp = b._fft_frame(windowed)
            power    = b._power(re, im)
            mel_e    = b._filterbank(power)
            log_pre  = b._log_compress(mel_e)
            out[:, f] = (b._bfp_compensate(log_pre, bfpexp)
                         if self.bfp_compensate else log_pre)
        return out

    def extract_float(self, pcm: np.ndarray) -> np.ndarray:
        """Same as extract() but returns float32 log2 values."""
        return self.extract(pcm).astype(np.float32) / (1 << Q_FRAC)


if __name__ == "__main__":
    import time

    print("=== GoldenExtractor (STFFT only) ===")
    ext = GoldenExtractor()
    t   = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    chirp = (np.sin(2 * np.pi * (200 * t + (7000-200)/2 * t**2)) * SAMPLE_MAX
             ).astype(np.int32)

    t0    = time.perf_counter()
    feats = ext.extract(chirp)
    dt    = time.perf_counter() - t0
    feats_f = feats.astype(np.float32) / (1 << Q_FRAC)
    print("  Input : %d samples (%.2fs)" % (len(chirp), len(chirp)/SAMPLE_RATE))
    print("  Output: %s  dtype=%s" % (feats.shape, feats.dtype))
    print("  Range : [%.3f, %.3f] log2" % (feats_f.min(), feats_f.max()))
    print("  Time  : %.1f ms" % (dt * 1000))

    print()
    print("=== FullPipelineGoldenExtractor (CIC + compFIR + STFFT) ===")
    full = FullPipelineGoldenExtractor()
    # Use a shorter chirp to keep runtime reasonable in smoke test
    short_chirp = (np.sin(2 * np.pi * (200 * t[:7500] +
                   (7000-200) / (2 * 0.47) * t[:7500]**2)) * SAMPLE_MAX
                  ).astype(np.int32)

    t0      = time.perf_counter()
    feats_f = full.extract_float(short_chirp)
    dt      = time.perf_counter() - t0
    print("  Input : %d PCM -> %d PDM bits" % (len(short_chirp), len(short_chirp)*63))
    print("  Output: %s  dtype=%s" % (feats_f.shape, feats_f.dtype))
    print("  Range : [%.3f, %.3f] log2" % (feats_f.min(), feats_f.max()))
    print("  Time  : %.1f ms" % (dt * 1000))

    if feats_f.max() > 5.0:
        print("  PASS -- signal content detected")
    else:
        print("  WARN -- very low energy, check CIC/FIR truncation")