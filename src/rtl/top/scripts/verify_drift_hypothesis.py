"""Verify the RTL-sample-drop hypothesis.

Hypothesis: the RTL's stfft drops samples during each R2FFT's ST_RUN_FFT +
ST_DONE phase, causing an alternating 129/148 effective hop instead of
the spec-correct 128.  Consequence: RTL frame k does NOT process samples
[k*128, k*128+256); it processes [sync_k - LATENCY - 256, sync_k - LATENCY).

Proof: if we feed the golden the SAME 256-sample windows that the RTL is
actually processing (derived from rtl_sync_sample_counts.npy), the
golden's FFT bins should match the RTL's bit-exactly.
"""
import os, sys
import numpy as np

_TOP = os.path.dirname(os.path.abspath(__file__))
_ML  = os.path.normpath(os.path.join(_TOP, "..", "..", "ml"))
sys.path.insert(0, _ML)

from golden_model import GoldenExtractor, SAMPLE_RATE, SAMPLE_W, N_FFT, Q_FRAC

SAMPLE_MAX = (1 << (SAMPLE_W - 1)) - 1
N_SAMPLES = 7500

# Latency from buffer-fill to fft_sync_rr:
#   R2FFT's FFT+DMA-start takes ~N_FFT/PL_depth cycles full-rate, then the
#   sample_counter has advanced by that / CE_EVERY sample-times.  Plus 2
#   register delays for fft_sync_rr vs o_fft_sync.  Empirical: 17–18 samples.
# First measured sync is at absolute sample 273 for a buffer of [0..255], so
# sync_latency = 273 - 256 = 17 samples exactly for A channel, may differ for
# B on the first emission due to b_armed timing.  Use 17.
SYNC_LATENCY = 17


def make_chirp(n):
    dur = n / SAMPLE_RATE
    t = np.arange(n) / SAMPLE_RATE
    phase = 2 * np.pi * (200 * t + (7000 - 200) / (2 * dur) * t**2)
    return (np.sin(phase) * SAMPLE_MAX).astype(np.int32)


def main():
    # Inputs: RTL outputs
    rtl_re = np.load(os.path.join(_TOP, "rtl_fft_re.npy"))
    rtl_im = np.load(os.path.join(_TOP, "rtl_fft_im.npy"))
    rtl_bfp = np.load(os.path.join(_TOP, "rtl_bfpexps.npy"))
    sync_cnts = np.load(os.path.join(_TOP, "rtl_sync_sample_counts.npy"))

    Nrtl = rtl_re.shape[0]
    assert Nrtl == len(sync_cnts), f"{Nrtl} != {len(sync_cnts)}"
    print(f"Loaded {Nrtl} RTL frames with sync sample counts.")
    print(f"First few sync_cnts: {list(sync_cnts[:10])}")
    print(f"Diffs: {list(np.diff(sync_cnts[:10]))}  (expect alt 129/148)")

    # Generate the chirp (the RTL was driven with this exact stream).
    samples = make_chirp(N_SAMPLES)

    ext = GoldenExtractor()

    # For each RTL frame, derive the 256-sample window it actually processed
    # and run that through the golden's FFT.
    gld_re = np.zeros_like(rtl_re)
    gld_im = np.zeros_like(rtl_im)
    gld_bfp = np.zeros(Nrtl, dtype=np.int32)
    windows = []

    for k in range(Nrtl):
        frame_end_excl = int(sync_cnts[k]) - SYNC_LATENCY   # one past the last sample
        frame_start    = frame_end_excl - N_FFT
        if frame_start < 0 or frame_end_excl > N_SAMPLES:
            # Out of range — pad with zeros
            win_input = np.zeros(N_FFT, dtype=np.int32)
            if frame_start < 0:
                lo = max(0, frame_start)
                win_input[lo - frame_start:] = samples[lo:frame_end_excl].astype(np.int32)
            else:
                hi = min(N_SAMPLES, frame_end_excl)
                win_input[:hi - frame_start] = samples[frame_start:hi].astype(np.int32)
        else:
            win_input = samples[frame_start:frame_end_excl].astype(np.int32)
        windows.append((frame_start, frame_end_excl))

        win_input = np.clip(win_input, -SAMPLE_MAX-1, SAMPLE_MAX)
        w = ext._window_frame(win_input)
        re_u, im_u, bfp = ext._fft_frame(w)
        re_s = re_u.astype(np.int64)
        im_s = im_u.astype(np.int64)
        re_s = np.where(re_s >= 0x8000, re_s - 0x10000, re_s)
        im_s = np.where(im_s >= 0x8000, im_s - 0x10000, im_s)
        gld_re[k, :] = re_s.astype(np.int32)
        gld_im[k, :] = im_s.astype(np.int32)
        gld_bfp[k] = bfp

    print(f"\nFirst few golden window ranges: {windows[:6]}")

    # Bit-exact comparison
    re_eq = (gld_re == rtl_re).sum()
    im_eq = (gld_im == rtl_im).sum()
    total = Nrtl * 129
    print(f"\n[Drift-corrected golden vs RTL, bit-exact]")
    print(f"  re: {re_eq}/{total} ({re_eq/total*100:.2f}%)")
    print(f"  im: {im_eq}/{total} ({im_eq/total*100:.2f}%)")

    # Bfpexp comparison
    bfp_eq = (gld_bfp[1:] == rtl_bfp[1:]).sum()  # skip frame 0 (startup)
    print(f"  bfpexp (excluding f0 startup): {bfp_eq}/{Nrtl-1}")

    # Peak bin comparison
    rtl_mag = np.sqrt(rtl_re.astype(np.float64)**2 + rtl_im.astype(np.float64)**2)
    gld_mag = np.sqrt(gld_re.astype(np.float64)**2 + gld_im.astype(np.float64)**2)
    peak_same = sum(
        int(np.argmax(rtl_mag[k])) == int(np.argmax(gld_mag[k]))
        for k in range(1, Nrtl)  # skip f0 (startup)
    )
    print(f"  peak-bin agreement (excluding f0): {peak_same}/{Nrtl-1}")

    # Show one tricky frame
    print(f"\n[Frame 20 bin-by-bin after drift correction]")
    print(f"{'bin':>4}  {'rtl_re':>8} {'rtl_im':>8}  {'gld_re':>8} {'gld_im':>8}  "
          f"{'match?':>8}")
    for k in list(range(40, 55)):
        m = "✓" if (rtl_re[20,k]==gld_re[20,k] and rtl_im[20,k]==gld_im[20,k]) else "×"
        print(f"  {k:3d}  {rtl_re[20,k]:8d} {rtl_im[20,k]:8d}  "
              f"{gld_re[20,k]:8d} {gld_im[20,k]:8d}  {m:>8}")


if __name__ == "__main__":
    main()
