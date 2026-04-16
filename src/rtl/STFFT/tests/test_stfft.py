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

async def feed_samples(dut, samples, warmup_frames=0):
    for s in samples:
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

    # 4-cycle pipeline latency:
    # dmaact→dmaact_r (1) + R2FFT internal DMA read (1) + SRAM (1) + dmadr→o_fft_result (1)
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
    cocotb.start_soon(feed_samples(dut, samples, warmup_frames=0))
    results_raw, bfpexp = await collect_results(dut)
    results_scaled = apply_bfp(results_raw, bfpexp)
    mags = [abs(complex(r, i)) for r, i in results_scaled]
    peak_bin = int(np.argmax(mags))
    dut._log.info(f"Peak bin: {peak_bin}, bfpexp: {bfpexp}")
    dut._log.info(f"Top 5: {sorted(enumerate(mags), key=lambda x: -x[1])[:5]}")
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

@cocotb.test()
async def test_ram_conflicts(dut):
    """Check if R2FFT ever asserts ract and wact simultaneously."""
    cocotb.start_soon(Clock(dut.i_clk, 10, units="ns").start())
    await reset_dut(dut)

    samples = make_tone(bin_k=8)
    cocotb.start_soon(feed_samples(dut, samples, warmup_frames=0))

    conflicts_ram0 = 0
    conflicts_ram1 = 0

    for _ in range(100000):
        await RisingEdge(dut.i_clk)

        r0 = dut.u_ram0.ract.value
        w0 = dut.u_ram0.wact.value
        r1 = dut.u_ram1.ract.value
        w1 = dut.u_ram1.wact.value

        if r0 == 1 and w0 == 1:
            conflicts_ram0 += 1
        if r1 == 1 and w1 == 1:
            conflicts_ram1 += 1

        if dut.o_fft_sync.value == 1:
            break

    dut._log.info(f"RAM0 simultaneous ract+wact conflicts: {conflicts_ram0}")
    dut._log.info(f"RAM1 simultaneous ract+wact conflicts: {conflicts_ram1}")

    if conflicts_ram0 == 0 and conflicts_ram1 == 0:
        dut._log.info("No conflicts — single-port SRAMs will work for ASIC")
    else:
        dut._log.info("Conflicts exist — dual-port or time-multiplex needed")

@cocotb.test()
async def test_ram_access_pattern(dut):
    """
    Check if R2FFT reads from ram0 while writing to ram1 and vice versa.
    If true, each bank is naturally single-port compatible.
    """
    cocotb.start_soon(Clock(dut.i_clk, 10, units="ns").start())
    await reset_dut(dut)

    samples = make_tone(bin_k=8)
    cocotb.start_soon(feed_samples(dut, samples, warmup_frames=0))

    # Count cross-bank vs same-bank simultaneous access
    same_bank_conflicts = 0   # read ram0 + write ram0 simultaneously
    cross_bank_ops      = 0   # read ram0 + write ram1 (or vice versa)

    for _ in range(100000):
        await RisingEdge(dut.i_clk)

        r0 = int(dut.u_ram0.ract.value)
        w0 = int(dut.u_ram0.wact.value)
        r1 = int(dut.u_ram1.ract.value)
        w1 = int(dut.u_ram1.wact.value)

        # Same-bank conflict: read and write to same physical bank
        if (r0 and w0) or (r1 and w1):
            same_bank_conflicts += 1

        # Cross-bank: read one, write the other
        if (r0 and w1) or (r1 and w0):
            cross_bank_ops += 1

        if dut.o_fft_sync.value == 1:
            break

    dut._log.info(f"Same-bank conflicts (need dual-port): {same_bank_conflicts}")
    dut._log.info(f"Cross-bank ops (single-port friendly): {cross_bank_ops}")

    if same_bank_conflicts == 0:
        dut._log.info("GOOD: each bank is naturally single-port compatible")
        dut._log.info("Solution: keep ram0 and ram1 as separate single-port SRAMs")
    else:
        dut._log.info("PROBLEM: same-bank simultaneous read+write exists")
        dut._log.info(f"Conflicts per total cross-bank: {same_bank_conflicts}/{cross_bank_ops}")

@cocotb.test()
async def test_next_stage_count(dut):
    """Verify next_stage fires exactly FFT_N = 8 times per frame."""
    cocotb.start_soon(Clock(dut.i_clk, 10, units="ns").start())
    await reset_dut(dut)

    samples = make_tone(bin_k=8)
    cocotb.start_soon(feed_samples(dut, samples, warmup_frames=0))

    count = 0
    for _ in range(100000):
        await RisingEdge(dut.i_clk)
        if dut.u_ram0.next_stage.value == 1:
            count += 1
        if dut.o_fft_sync.value == 1:
            break

    dut._log.info(f"next_stage fired {count} times (expected 8 for 256-point FFT)")
    assert count == 8, f"Expected 8 stage boundaries, got {count}"