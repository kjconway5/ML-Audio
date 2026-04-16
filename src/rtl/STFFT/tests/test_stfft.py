# tb/fft/test_stfft.py
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, with_timeout, ClockCycles, Timer, ReadOnly
import numpy as np
from scipy.signal.windows import hann


CE_EVERY     = 20       # clocks between valid_i pulses (CKPCE=3 + logmel margin)

def bit_reverse(x, bits=8):
    y = 0
    for i in range(bits):
        if x & (1 << i):
            y |= 1 << (bits - 1 - i)
    return y

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

async def reset_dut(dut, cycles=20):  # increase to 20 to fully flush pipelines
    dut.i_reset.value = 1
    dut.i_ce.value    = 0
    dut.i_sample.value = 0
    for _ in range(cycles):
        await RisingEdge(dut.i_clk)
    dut.i_reset.value = 0
    await RisingEdge(dut.i_clk)

async def feed_samples(dut, samples, warmup_frames=1):
    """
    Always feed warmup_frames extra frames to flush windowfn's first_block.
    CE_EVERY spacing ensures alt_ce never collides with primary_ce.
    """
    all_samples = list(samples) * (warmup_frames + 1)
    for s in all_samples:
        dut.i_ce.value = 1
        dut.i_sample.value = int(s) & 0xFFFF
        await RisingEdge(dut.i_clk)
        dut.i_ce.value = 0
        await ClockCycles(dut.i_clk, CE_EVERY - 1)

async def collect_results(dut, N=256, timeout_cycles=200000):
    OW = 16

    sync_event = cocotb.triggers.Event()

    async def watch_sync():
        while True:
            await RisingEdge(dut.i_clk)
            if dut.o_fft_sync.value == 1:
                sync_event.set()
                return

    watcher = cocotb.start_soon(watch_sync())

    try:
        await with_timeout(sync_event.wait(), timeout_cycles * 10, 'ns')
    except cocotb.result.SimTimeoutError:
        watcher.kill()
        raise TimeoutError("o_fft_sync never asserted")

    # Pipeline latency from dma_addr to o_fft_result:
    # 1 cycle: dmaact→dmaact_r, dmaa→dmaa_r (registered in stfft)
    # 1 cycle: ract→ract_r, ra→ra_r (registered in stfft)  
    # 1 cycle: SRAM read
    # 1 cycle: dmadr_w→dmadr_r (registered in stfft)
    # = 4 cycles total
    await ClockCycles(dut.i_clk, 4)

    results = []
    for _ in range(N):
        raw = int(dut.o_fft_result.value)
        re_raw = (raw >> OW) & ((1 << OW) - 1)
        im_raw =  raw        & ((1 << OW) - 1)
        if re_raw & (1 << (OW-1)): re_raw -= (1 << OW)
        if im_raw & (1 << (OW-1)): im_raw -= (1 << OW)
        results.append((re_raw, im_raw))
        await RisingEdge(dut.i_clk)

    bfpexp = int(dut.o_bfpexp.value.signed_integer)
    return results, bfpexp

# Tests

@cocotb.test()
async def test_tone_bin_location(dut):
    """Quick check: pure tone at bin 8 should peak at bin 8."""
    cocotb.start_soon(Clock(dut.i_clk, 10, units="ns").start())
    await reset_dut(dut)
    samples = make_tone(bin_k=8)
    cocotb.start_soon(feed_samples(dut, samples, warmup_frames=1))
    results_raw, bfpexp = await collect_results(dut)
    results_scaled = apply_bfp(results_raw, bfpexp)
    mags = [abs(complex(r, i)) for r, i in results_scaled]
    peak_bin = int(np.argmax(mags))
    dut._log.info(f"Peak bin: {peak_bin}, bfpexp: {bfpexp}")
    dut._log.info(f"Top 5 bins: {sorted(enumerate(mags), key=lambda x: -x[1])[:5]}")
    assert peak_bin == 8, f"Expected peak at bin 8, got bin {peak_bin}"

@cocotb.test()
async def test_pure_tone_snr(dut):
    cocotb.start_soon(Clock(dut.i_clk, 10, units="ns").start())
    await reset_dut(dut)
    samples = make_tone(bin_k=8)
    ref_fft = numpy_reference(samples)
    cocotb.start_soon(feed_samples(dut, samples, warmup_frames=1))
    results_raw, bfpexp = await collect_results(dut)
    results_scaled = apply_bfp(results_raw, bfpexp)
    snr = snr_db(results_scaled, ref_fft)
    dut._log.info(f"Pure tone SNR: {snr:.1f} dB  (bfpexp={bfpexp})")
    assert snr > 50.0, f"SNR too low: {snr:.1f} dB"

@cocotb.test()
async def test_dc_input(dut):
    cocotb.start_soon(Clock(dut.i_clk, 10, units="ns").start())
    await reset_dut(dut)
    samples = np.full(256, 4000, dtype=np.int16)
    ref_fft = numpy_reference(samples)
    cocotb.start_soon(feed_samples(dut, samples, warmup_frames=1))
    results_raw, bfpexp = await collect_results(dut)
    results_scaled = apply_bfp(results_raw, bfpexp)
    mags = [abs(complex(r, i)) for r, i in results_scaled]
    peak_bin = int(np.argmax(mags))
    assert peak_bin == 0, f"DC peak expected at bin 0, got bin {peak_bin}"

@cocotb.test()
async def test_nyquist_tone(dut):
    cocotb.start_soon(Clock(dut.i_clk, 10, units="ns").start())
    await reset_dut(dut)
    samples = make_tone(bin_k=128)
    ref_fft = numpy_reference(samples)
    cocotb.start_soon(feed_samples(dut, samples, warmup_frames=1))
    results_raw, bfpexp = await collect_results(dut)
    results_scaled = apply_bfp(results_raw, bfpexp)
    snr = snr_db(results_scaled, ref_fft)
    dut._log.info(f"Nyquist tone SNR: {snr:.1f} dB")
    assert snr > 50.0

@cocotb.test()
async def test_zero_input(dut):
    cocotb.start_soon(Clock(dut.i_clk, 10, units="ns").start())
    await reset_dut(dut)
    samples = np.zeros(256, dtype=np.int16)
    cocotb.start_soon(feed_samples(dut, samples, warmup_frames=1))
    results_raw, bfpexp = await collect_results(dut)
    for k, (r, i) in enumerate(results_raw):
        assert r == 0 and i == 0, \
            f"Zero input: non-zero output at bin {k}: ({r}, {i})"
        
@cocotb.test()
async def test_debug_fsm(dut):
    cocotb.start_soon(Clock(dut.i_clk, 10, units="ns").start())
    await reset_dut(dut)

    samples = make_tone(bin_k=8)
    
    # Feed TWO frames — windowfn suppresses first frame
    for frame in range(2):
        for i, s in enumerate(samples):
            dut.i_ce.value = 1
            dut.i_sample.value = int(s) & 0xFFFF
            await RisingEdge(dut.i_clk)
            dut.i_ce.value = 0
            await RisingEdge(dut.i_clk)  # gap for alt_ce

        dut._log.info(f"Frame {frame} done, win_ce={dut.win_ce_o.value}")

    # Now wait for sync
    for i in range(5000):
        await RisingEdge(dut.i_clk)
        if dut.o_fft_sync.value == 1:
            dut._log.info(f"SUCCESS: sync at cycle {i}")
            return
        if i % 500 == 0:
            dut._log.info(
                f"cycle {i}: win_ce={dut.win_ce_o.value} "
                f"status={dut.u_r2fft.status.value} "
                f"done={dut.u_r2fft.done.value}"
            )

    assert False, "Never got sync"
@cocotb.test()
async def test_window_coefficients(dut):
    """Check if window coefficients are loaded correctly."""

    cocotb.start_soon(Clock(dut.i_clk, 10, units="ns").start())
    await reset_dut(dut)

    u_win = dut.u_win

    # Check coefficient memory 
    dut._log.info("Checking window coefficients from memory:")

    for i in range(5):
        try:
            if hasattr(u_win, "cmem"):
                coeff = u_win.cmem[i].value
                dut._log.info(f"cmem[{i}] = {coeff}")
            else:
                dut._log.info(f"cmem not directly accessible at index {i}")
        except Exception as e:
            dut._log.info(f"Cannot read cmem[{i}]: {e}")

    # Feed samples using YOUR correct CE model
    samples = [1000] * 256

    cocotb.start_soon(feed_samples(dut, samples))

    # Monitor internal window behavior
    for i in range(100000):
        await RisingEdge(dut.i_clk)
        # SUCCESS CONDITION
        if int(dut.win_ce_o.value) == 1:
            dut._log.info(f"SUCCESS: win_ce_o asserted at cycle {i}")

            # Optional sanity check: window actually producing data
            try:
                if hasattr(u_win, "product"):
                    if int(u_win.product.value) != 0:
                        dut._log.info(f"Product active: {u_win.product.value}")
            except:
                pass

            return

    # FAILURE HANDLING
    dut._log.error("win_ce_o never asserted - window not producing output")

    # Debug dump
    dut._log.info("Final state dump:")
    dut._log.info(f"i_ce={dut.i_ce.value}, alt_ce={dut.alt_ce.value}")
    dut._log.info(f"win_ce_o={dut.win_ce_o.value}, inner o_ce={u_win.o_ce.value}")

    if hasattr(u_win, "cmem"):
        zero_count = 0
        for i in range(10):
            if int(u_win.cmem[i].value) == 0:
                zero_count += 1
        dut._log.info(f"cmem zeros in first 10 taps: {zero_count}/10")

@cocotb.test()
async def test_debug_window(dut):
    """Debug windowing stage."""
    cocotb.start_soon(Clock(dut.i_clk, 10, units="ns").start())
    await reset_dut(dut)

    samples = make_tone(bin_k=8)

    for i, s in enumerate(samples):
        dut.i_ce.value = 1
        dut.i_sample.value = int(s) & 0xFFFF
        await RisingEdge(dut.i_clk)

        if i < 10:
            dut._log.info(
                f"sample {i}: i_ce={dut.i_ce.value} "
                f"win_ce={dut.win_ce_o.value} "
                f"i_sample={dut.i_sample.value} "
                f"alt_delay={dut.alt_delay.value} "
                f"streamBufferFull={dut.u_r2fft.streamBufferFull.value} "
                f"sact_istream={dut.u_r2fft.sact_istream.value}"
            )

    dut.i_ce.value = 0
    # Wait for window to flush
    for i in range(100000):
        await RisingEdge(dut.i_clk)
        if dut.win_ce_o.value == 1:
            dut._log.info(f"win_ce asserted at cycle {i}")
            break
    else:
        dut._log.info("win_ce NEVER asserted - window not producing output")

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

        # Small gap between frames 
        for _ in range(20):
            await RisingEdge(dut.i_clk)