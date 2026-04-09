# tb/fft/test_stfft.py
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, with_timeout
import numpy as np
from scipy.signal.windows import hann

# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def make_tone(bin_k, N=256, amplitude=8191):
    """Pure tone at FFT bin k, 14-bit signed."""
    t = np.arange(N)
    return (amplitude * np.cos(2 * np.pi * bin_k * t / N)).astype(np.int16)

def numpy_reference(samples, N=256):
    """
    Compute reference FFT matching your hardware pipeline:
    Hanning window → FFT → return complex array.
    """
    window   = hann(N, sym=False)
    windowed = samples.astype(np.float64) * window
    return np.fft.fft(windowed)

def apply_bfp(results_raw, bfpexp):
    """
    Scale raw hardware output by BFP exponent.
    true_value = stored × 2^bfpexp
    """
    scale = 2.0 ** int(bfpexp)
    return [(r * scale, i * scale) for r, i in results_raw]

def snr_db(hw_vals, ref_fft):
    """Compute SNR between hardware output and numpy reference."""
    hw  = np.array([complex(r, i) for r, i in hw_vals])
    ref = ref_fft / np.max(np.abs(ref_fft))   # normalize reference
    hw  = hw      / np.max(np.abs(hw) + 1e-9) # normalize hardware
    noise  = hw - ref
    signal_power = np.mean(np.abs(ref)**2)
    noise_power  = np.mean(np.abs(noise)**2)
    return 10 * np.log10(signal_power / (noise_power + 1e-12))

async def reset_dut(dut, cycles=10):
    dut.i_reset.value = 1
    dut.i_ce.value    = 0
    dut.i_sample.value = 0
    for _ in range(cycles):
        await RisingEdge(dut.i_clk)
    dut.i_reset.value = 0
    await RisingEdge(dut.i_clk)

async def feed_samples(dut, samples):
    """Stream samples into DUT one per clock with i_ce."""
    for s in samples:
        await FallingEdge(dut.i_clk)
        dut.i_ce.value     = 1
        dut.i_sample.value = int(s) & 0x3FFF   # 14-bit mask
    await FallingEdge(dut.i_clk)
    dut.i_ce.value = 0

async def collect_results(dut, N=256, timeout_cycles=50000):
    """
    Wait for o_fft_sync, then collect N output bins.
    Returns (results_raw, bfpexp) where results_raw is
    a list of (real, imag) signed integers.
    """
    OW = 18

    # Wait for sync pulse
    for _ in range(timeout_cycles):
        await RisingEdge(dut.i_clk)
        if dut.o_fft_sync.value == 1:
            break
    else:
        raise TimeoutError("o_fft_sync never asserted")

    # Collect N bins
    results = []
    for _ in range(N):
        raw = int(dut.o_fft_result.value)
        # Unpack two OW-bit signed fields
        re_raw = (raw >> OW) & ((1 << OW) - 1)
        im_raw =  raw        & ((1 << OW) - 1)
        # Sign extend
        if re_raw & (1 << (OW-1)): re_raw -= (1 << OW)
        if im_raw & (1 << (OW-1)): im_raw -= (1 << OW)
        results.append((re_raw, im_raw))
        await RisingEdge(dut.i_clk)

    bfpexp = int(dut.o_bfpexp.value.signed_integer)
    return results, bfpexp

# ----------------------------------------------------------------
# Tests
# ----------------------------------------------------------------

@cocotb.test()
async def test_pure_tone_snr(dut):
    """
    Feed a pure tone at bin 8, verify SNR > 50dB against numpy reference.
    50dB is conservative for 14-bit input — you should see ~60dB+.
    """
    cocotb.start_soon(Clock(dut.i_clk, 10, units="ns").start())
    await reset_dut(dut)

    samples = make_tone(bin_k=8)
    ref_fft = numpy_reference(samples)

    cocotb.start_soon(feed_samples(dut, samples))
    results_raw, bfpexp = await collect_results(dut)
    results_scaled = apply_bfp(results_raw, bfpexp)

    snr = snr_db(results_scaled, ref_fft)
    dut._log.info(f"Pure tone SNR: {snr:.1f} dB  (bfpexp={bfpexp})")
    assert snr > 50.0, f"SNR too low: {snr:.1f} dB"

@cocotb.test()
async def test_dc_input(dut):
    """DC input — all energy should be in bin 0."""
    cocotb.start_soon(Clock(dut.i_clk, 10, units="ns").start())
    await reset_dut(dut)

    samples = np.full(256, 4000, dtype=np.int16)
    ref_fft = numpy_reference(samples)

    cocotb.start_soon(feed_samples(dut, samples))
    results_raw, bfpexp = await collect_results(dut)
    results_scaled = apply_bfp(results_raw, bfpexp)

    # Bin 0 magnitude should dominate
    mags = [abs(complex(r, i)) for r, i in results_scaled]
    peak_bin = int(np.argmax(mags))
    assert peak_bin == 0, f"DC peak expected at bin 0, got bin {peak_bin}"

@cocotb.test()
async def test_nyquist_tone(dut):
    """Tone at bin N/2 — tests highest frequency path."""
    cocotb.start_soon(Clock(dut.i_clk, 10, units="ns").start())
    await reset_dut(dut)

    samples = make_tone(bin_k=128)
    ref_fft = numpy_reference(samples)

    cocotb.start_soon(feed_samples(dut, samples))
    results_raw, bfpexp = await collect_results(dut)
    results_scaled = apply_bfp(results_raw, bfpexp)

    snr = snr_db(results_scaled, ref_fft)
    dut._log.info(f"Nyquist tone SNR: {snr:.1f} dB")
    assert snr > 50.0

@cocotb.test()
async def test_zero_input(dut):
    """All-zero input should produce all-zero output, no spurious energy."""
    cocotb.start_soon(Clock(dut.i_clk, 10, units="ns").start())
    await reset_dut(dut)

    samples = np.zeros(256, dtype=np.int16)
    cocotb.start_soon(feed_samples(dut, samples))
    results_raw, bfpexp = await collect_results(dut)

    for k, (r, i) in enumerate(results_raw):
        assert r == 0 and i == 0, \
            f"Zero input: non-zero output at bin {k}: ({r}, {i})"

@cocotb.test()
async def test_consecutive_frames(dut):
    """
    Feed two consecutive frames, verify both produce valid output.
    This exercises the autorun re-trigger path in R2FFT's FSM.
    """
    cocotb.start_soon(Clock(dut.i_clk, 10, units="ns").start())
    await reset_dut(dut)

    for frame_idx, bin_k in enumerate([8, 32]):
        samples = make_tone(bin_k=bin_k)
        ref_fft = numpy_reference(samples)

        cocotb.start_soon(feed_samples(dut, samples))
        results_raw, bfpexp = await collect_results(dut)
        results_scaled = apply_bfp(results_raw, bfpexp)

        snr = snr_db(results_scaled, ref_fft)
        dut._log.info(f"Frame {frame_idx} (bin {bin_k}): SNR={snr:.1f} dB")
        assert snr > 50.0, \
            f"Frame {frame_idx} SNR too low: {snr:.1f} dB"

        # Small gap between frames (simulates real audio stream)
        for _ in range(20):
            await RisingEdge(dut.i_clk)