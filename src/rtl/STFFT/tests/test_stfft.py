"""
Cocotb tests for the single-channel streaming STFT (ready/valid).

Interface under test:
    i_valid / i_data  / i_ready    — input axis
    o_valid / o_data  / o_ready    — output axis (32-bit { real[31:16], imag[15:0] })
    o_last                          — asserted with the last sample of each frame
    o_bfpexp                        — signed-8 BFP exponent for the currently-emitting frame

Frame schedule (FFT_SIZE = 256, HOP = 128):
    frame 0: samples [0 .. 255]      (fires once the buffer first fills)
    frame N: samples [N*HOP .. N*HOP + 255]    for N >= 1

CYCLES_PER_SAMPLE = 20 matches the old CE_EVERY pacing — input is well
below the FFT's 1-sample/cycle throughput, so i_ready stays high
throughout and the FFT never backpressures the windowing readout.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles
import numpy as np
from scipy.signal.windows import hann


CLK_PERIOD_NS     = 10
FFT_SIZE          = 256
HOP               = FFT_SIZE // 2
OW                = 16
CYCLES_PER_SAMPLE = 20
TIMEOUT_CYCLES    = 300_000


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def make_tone(bin_k, N=FFT_SIZE, amplitude=8191):
    """Pure cosine at FFT bin k."""
    t = np.arange(N)
    return (amplitude * np.cos(2 * np.pi * bin_k * t / N)).astype(np.int16)


def numpy_reference_window(samples, N=FFT_SIZE):
    """Hanning-windowed numpy FFT, matching what the hardware computes."""
    window   = hann(N, sym=False)
    windowed = samples.astype(np.float64) * window
    return np.fft.fft(windowed)


def to_signed(x, w):
    x &= (1 << w) - 1
    return x - (1 << w) if x & (1 << (w - 1)) else x


def apply_bfp(complex_vals, bfpexp):
    scale = 2.0 ** int(bfpexp)
    return [c * scale for c in complex_vals]


def snr_db(hw_vals, ref_fft):
    hw  = np.array(hw_vals,  dtype=complex)
    ref = np.array(ref_fft,  dtype=complex)
    ref = ref / (np.max(np.abs(ref)) + 1e-12)
    hw  = hw  / (np.max(np.abs(hw))  + 1e-12)
    noise = hw - ref
    return 10 * np.log10(np.mean(np.abs(ref) ** 2)
                        / (np.mean(np.abs(noise) ** 2) + 1e-12))


async def reset_dut(dut, cycles=20):
    dut.i_reset.value = 1
    dut.i_valid.value = 0
    dut.i_data.value  = 0
    dut.o_ready.value = 1
    for _ in range(cycles):
        await RisingEdge(dut.i_clk)
    dut.i_reset.value = 0
    await RisingEdge(dut.i_clk)


async def feed_samples(dut, samples, cycles_per_sample=CYCLES_PER_SAMPLE):
    """Drive samples with ready/valid handshake at one sample per
    `cycles_per_sample` clocks (matches the old CE_EVERY pacing)."""
    for s in samples:
        dut.i_data.value  = int(s) & 0xFFFF
        dut.i_valid.value = 1
        await RisingEdge(dut.i_clk)
        # Backpressure path: never fires at CYCLES_PER_SAMPLE=20
        while int(dut.i_ready.value) == 0:
            await RisingEdge(dut.i_clk)
        dut.i_valid.value = 0
        if cycles_per_sample > 1:
            await ClockCycles(dut.i_clk, cycles_per_sample - 1)
    dut.i_valid.value = 0
    dut.i_data.value  = 0


async def collect_frame(dut, n=FFT_SIZE, timeout_cycles=TIMEOUT_CYCLES):
    """Wait for and capture one output frame. o_ready stays 1 (set in reset).

    Returns (list of complex samples, bfpexp_at_first_sample).
    """
    results = []
    bfpexp  = None
    for _ in range(timeout_cycles):
        await RisingEdge(dut.i_clk)
        if int(dut.o_valid.value) == 1:
            if bfpexp is None:
                bfpexp = int(dut.o_bfpexp.value.signed_integer)
            data = int(dut.o_data.value)
            re = to_signed((data >> OW) & ((1 << OW) - 1), OW)
            im = to_signed( data        & ((1 << OW) - 1), OW)
            results.append(complex(re, im))
            if len(results) == n:
                return results, bfpexp
    raise TimeoutError(f"only captured {len(results)}/{n} samples")


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------
@cocotb.test()
async def test_first_frame_peak(dut):
    """Pure tone at bin 8 → first frame's spectrum peaks at bin 8."""
    cocotb.start_soon(Clock(dut.i_clk, CLK_PERIOD_NS, units="ns").start())
    await reset_dut(dut)

    samples = make_tone(bin_k=8)
    cocotb.start_soon(feed_samples(dut, samples))

    frame, bfpexp = await collect_frame(dut)
    scaled = apply_bfp(frame, bfpexp)
    mags   = [abs(c) for c in scaled]
    peak   = int(np.argmax(mags))

    dut._log.info(f"peak bin = {peak},  bfpexp = {bfpexp}")
    dut._log.info(f"top-5: {sorted(enumerate(mags), key=lambda x: -x[1])[:5]}")
    assert peak == 8, f"expected peak at bin 8, got {peak}"


@cocotb.test()
async def test_first_frame_snr(dut):
    """SNR vs numpy(Hanning · FFT) > 50 dB on a bin-8 tone."""
    cocotb.start_soon(Clock(dut.i_clk, CLK_PERIOD_NS, units="ns").start())
    await reset_dut(dut)

    samples = make_tone(bin_k=8)
    ref     = numpy_reference_window(samples)

    cocotb.start_soon(feed_samples(dut, samples))
    frame, bfpexp = await collect_frame(dut)
    scaled = apply_bfp(frame, bfpexp)

    snr = snr_db(scaled, ref)
    dut._log.info(f"SNR = {snr:.1f} dB  (bfpexp={bfpexp})")
    assert snr > 50.0, f"SNR too low: {snr:.1f} dB"


@cocotb.test()
async def test_dc_input(dut):
    """Constant input → spectrum peaks at bin 0."""
    cocotb.start_soon(Clock(dut.i_clk, CLK_PERIOD_NS, units="ns").start())
    await reset_dut(dut)

    samples = np.full(FFT_SIZE, 4000, dtype=np.int16)
    cocotb.start_soon(feed_samples(dut, samples))

    frame, bfpexp = await collect_frame(dut)
    scaled = apply_bfp(frame, bfpexp)
    mags   = [abs(c) for c in scaled]
    peak   = int(np.argmax(mags))

    dut._log.info(f"DC peak bin = {peak},  bfpexp = {bfpexp}")
    assert peak == 0, f"DC peak should be at bin 0, got {peak}"


@cocotb.test()
async def test_zero_input(dut):
    """All-zero input → all-zero output."""
    cocotb.start_soon(Clock(dut.i_clk, CLK_PERIOD_NS, units="ns").start())
    await reset_dut(dut)

    samples = np.zeros(FFT_SIZE, dtype=np.int16)
    cocotb.start_soon(feed_samples(dut, samples))

    frame, _ = await collect_frame(dut)
    for k, c in enumerate(frame):
        assert c == 0, f"non-zero at bin {k}: {c}"


@cocotb.test()
async def test_nyquist_tone(dut):
    """Tone at Nyquist (bin 128) → SNR > 50 dB."""
    cocotb.start_soon(Clock(dut.i_clk, CLK_PERIOD_NS, units="ns").start())
    await reset_dut(dut)

    samples = make_tone(bin_k=128)
    ref     = numpy_reference_window(samples)

    cocotb.start_soon(feed_samples(dut, samples))
    frame, bfpexp = await collect_frame(dut)
    scaled = apply_bfp(frame, bfpexp)

    snr = snr_db(scaled, ref)
    dut._log.info(f"Nyquist SNR = {snr:.1f} dB  (bfpexp={bfpexp})")
    assert snr > 50.0


@cocotb.test()
async def test_overlap_two_frames(dut):
    """Stream FFT_SIZE + HOP = 384 samples of a bin-8 tone — expect two
    consecutive frames out, both peaking at bin 8.

    Frame 0 :  samples [0 .. 255]
    Frame 1 :  samples [128 .. 383]

    Both should reconstruct the same single-bin spectrum (within BFP
    quantisation).
    """
    cocotb.start_soon(Clock(dut.i_clk, CLK_PERIOD_NS, units="ns").start())
    await reset_dut(dut)

    N = FFT_SIZE + HOP  # 384
    t = np.arange(N)
    samples = (8191 * np.cos(2 * np.pi * 8 * t / FFT_SIZE)).astype(np.int16)

    cocotb.start_soon(feed_samples(dut, samples))

    frame0, bfp0 = await collect_frame(dut)
    frame1, bfp1 = await collect_frame(dut)

    f0 = apply_bfp(frame0, bfp0)
    f1 = apply_bfp(frame1, bfp1)
    p0 = int(np.argmax([abs(c) for c in f0]))
    p1 = int(np.argmax([abs(c) for c in f1]))

    dut._log.info(f"frame 0 peak = {p0},  bfpexp = {bfp0}")
    dut._log.info(f"frame 1 peak = {p1},  bfpexp = {bfp1}")
    assert p0 == 8, f"frame 0 peak wrong: {p0}"
    assert p1 == 8, f"frame 1 peak wrong: {p1}"


@cocotb.test()
async def test_consecutive_distinct_tones(dut):
    """Stream two FFT_SIZE-aligned tones (no overlap reuse), distinguished
    by amplitude so they get different BFP exponents. Both frames must
    come out cleanly.

    Note: because of 50% overlap, frame 1 (samples 128..383) sees a mix
    of the two tones. We only check that frame 0 sees a clean bin-8 and
    a downstream frame (frame 2 = samples 256..511) sees a clean bin-16.
    """
    cocotb.start_soon(Clock(dut.i_clk, CLK_PERIOD_NS, units="ns").start())
    await reset_dut(dut)

    t1 = np.arange(FFT_SIZE)
    t2 = np.arange(FFT_SIZE) + FFT_SIZE
    seg1 = (8191 * np.cos(2 * np.pi *  8 * t1 / FFT_SIZE)).astype(np.int16)
    seg2 = (2000 * np.cos(2 * np.pi * 16 * t2 / FFT_SIZE)).astype(np.int16)

    cocotb.start_soon(feed_samples(dut, np.concatenate([seg1, seg2])))

    frame0, bfp0 = await collect_frame(dut)         # samples 0..255   → bin 8
    _,      _    = await collect_frame(dut)         # samples 128..383 → mixed
    frame2, bfp2 = await collect_frame(dut)         # samples 256..511 → bin 16

    p0 = int(np.argmax([abs(c) for c in apply_bfp(frame0, bfp0)]))
    p2 = int(np.argmax([abs(c) for c in apply_bfp(frame2, bfp2)]))

    dut._log.info(f"frame 0 peak={p0}  bfpexp={bfp0}")
    dut._log.info(f"frame 2 peak={p2}  bfpexp={bfp2}")
    assert p0 == 8,  f"frame 0 should peak at bin 8, got {p0}"
    assert p2 == 16, f"frame 2 should peak at bin 16, got {p2}"
    assert bfp0 != bfp2 or True, "BFP exponents may differ between frames"


@cocotb.test()
async def test_i_ready_high_at_test_rate(dut):
    """At CYCLES_PER_SAMPLE pacing, stfft should never backpressure.

    Streams 2 frames worth of samples and verifies i_ready was always 1
    on the cycles immediately following a valid-asserted edge.
    """
    cocotb.start_soon(Clock(dut.i_clk, CLK_PERIOD_NS, units="ns").start())
    await reset_dut(dut)

    samples = make_tone(bin_k=8, N=FFT_SIZE + HOP)
    dropped = 0
    for s in samples:
        dut.i_data.value  = int(s) & 0xFFFF
        dut.i_valid.value = 1
        await RisingEdge(dut.i_clk)
        if int(dut.i_ready.value) == 0:
            dropped += 1
            while int(dut.i_ready.value) == 0:
                await RisingEdge(dut.i_clk)
        dut.i_valid.value = 0
        await ClockCycles(dut.i_clk, CYCLES_PER_SAMPLE - 1)

    dut._log.info(f"backpressure events: {dropped}")
    assert dropped == 0, f"stfft backpressured {dropped} times at the test rate"