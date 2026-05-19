"""
Cocotb smoke tests for the ping-pong / single-port FFT wrapper with
AXI-stream-style ready/valid interface.

Tests:
    - impulse at n=0       : flat spectrum, every bin = input[0]
    - constant             : single spike at bin 0
    - pure sine at bin 8   : energy at bins 8 and (N-8) only
    - two pipelined frames : back-to-back distinct sines, both must come
                             out cleanly with their own BFP exponents

Each single-frame test:
    1. resets, asserts o_ready (no backpressure)
    2. streams 256 samples via the i_valid / i_data / i_ready handshake
    3. captures 256 output transfers via o_valid / o_data (o_ready=1)
    4. multiplies by 2^bfpexp (block-floating-point recovery)
    5. compares against numpy.fft.fft of the (unquantised) input

bfpexp is read hierarchically as dut.u_fft.bfpexp; verilator preserves
internal signals with --trace --trace-structs.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
import numpy as np


CLK_PERIOD_NS  = 10
FFT_SIZE       = 256
TIMEOUT_CYCLES = 25000


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
async def reset_dut(dut, cycles=10):
    dut.i_reset.value  = 1
    dut.i_valid.value  = 0
    dut.i_data.value   = 0
    dut.o_ready.value  = 1     # consumer always ready (no backpressure in these tests)
    for _ in range(cycles):
        await RisingEdge(dut.i_clk)
    dut.i_reset.value = 0
    await RisingEdge(dut.i_clk)


async def stream_samples(dut, samples):
    """Drive samples with the ready-valid input handshake.

    Holds i_valid + i_data until i_ready is observed high at a clock edge,
    then advances to the next sample. In the tests below i_ready stays
    high for the whole stream, so each sample takes exactly one cycle.
    """
    for s in samples:
        dut.i_data.value  = int(s) & 0xFFFF
        dut.i_valid.value = 1
        await RisingEdge(dut.i_clk)
        # Backpressure path: only iterates if the FFT can't accept a sample
        # right now (e.g., 3+ frames with compute still running on both RAMs).
        while int(dut.i_ready.value) == 0:
            await RisingEdge(dut.i_clk)
    dut.i_valid.value = 0
    dut.i_data.value  = 0


def to_signed16(x):
    x &= 0xFFFF
    return x - 0x10000 if x >= 0x8000 else x


async def capture_output(dut, n_samples, timeout_cycles=TIMEOUT_CYCLES):
    """Capture n_samples output transfers (o_valid && o_ready, with o_ready=1).

    Returns (samples, bfpexps) where bfpexps[i] is the BFP exponent at the
    cycle sample i was emitted — useful for multi-frame captures.
    """
    out     = []
    bfpexps = []
    for _ in range(timeout_cycles):
        await RisingEdge(dut.i_clk)
        if int(dut.o_valid.value) == 1:
            data = int(dut.o_data.value)
            real = to_signed16((data >> 16) & 0xFFFF)
            imag = to_signed16(data & 0xFFFF)
            out.append(complex(real, imag))
            bfpexps.append(int(dut.u_fft.bfpexp.value.signed_integer))
            if len(out) == n_samples:
                return out, bfpexps
    raise TimeoutError(f"captured only {len(out)} / {n_samples} samples")


async def run_fft(dut, inputs):
    """Stream `inputs`, capture FFT_SIZE outputs concurrently, return
    (outputs, bfpexp_at_first_sample)."""
    streamer = cocotb.start_soon(stream_samples(dut, inputs))
    outputs, bfpexps = await capture_output(dut, FFT_SIZE)
    await streamer
    return outputs, bfpexps[0]


def compare(dut, measured, reference, label, abs_tol=200):
    max_err   = 0.0
    worst_bin = -1
    for k, (m, r) in enumerate(zip(measured, reference)):
        e = abs(m - r)
        if e > max_err:
            max_err   = e
            worst_bin = k
    dut._log.info(f"[{label}] max |err| = {max_err:.1f} at bin {worst_bin}  (tol={abs_tol})")
    assert max_err < abs_tol, (
        f"[{label}] max error {max_err:.1f} > {abs_tol}  (bin {worst_bin}: "
        f"measured={measured[worst_bin]}, reference={reference[worst_bin]})"
    )


# ----------------------------------------------------------------------------
# Test 1: impulse — flat spectrum
# ----------------------------------------------------------------------------
@cocotb.test()
async def test_impulse(dut):
    cocotb.start_soon(Clock(dut.i_clk, CLK_PERIOD_NS, units="ns").start())
    await reset_dut(dut)

    inp = np.zeros(FFT_SIZE, dtype=np.int32)
    inp[0] = 10000

    outputs, bfpexp = await run_fft(dut, inp)
    dut._log.info(f"[impulse] bfpexp = {bfpexp}")

    measured  = np.array(outputs, dtype=complex) * (2.0 ** bfpexp)
    reference = np.fft.fft(inp.astype(np.float64))
    compare(dut, measured, reference, "impulse", abs_tol=500)


# ----------------------------------------------------------------------------
# Test 2: constant — spike at bin 0
# ----------------------------------------------------------------------------
@cocotb.test()
async def test_constant(dut):
    cocotb.start_soon(Clock(dut.i_clk, CLK_PERIOD_NS, units="ns").start())
    await reset_dut(dut)

    inp = np.full(FFT_SIZE, 100, dtype=np.int32)

    outputs, bfpexp = await run_fft(dut, inp)
    dut._log.info(f"[constant] bfpexp = {bfpexp}")

    measured  = np.array(outputs, dtype=complex) * (2.0 ** bfpexp)
    reference = np.fft.fft(inp.astype(np.float64))
    compare(dut, measured, reference, "constant", abs_tol=500)


# ----------------------------------------------------------------------------
# Test 3: pure sine at bin 8 — energy at bins 8 and (N-8) only
# ----------------------------------------------------------------------------
@cocotb.test()
async def test_sine(dut):
    cocotb.start_soon(Clock(dut.i_clk, CLK_PERIOD_NS, units="ns").start())
    await reset_dut(dut)

    n   = np.arange(FFT_SIZE)
    inp = (5000 * np.sin(2 * np.pi * 8 * n / FFT_SIZE)).astype(np.int32)

    outputs, bfpexp = await run_fft(dut, inp)
    dut._log.info(f"[sine] bfpexp = {bfpexp}")

    measured  = np.array(outputs, dtype=complex) * (2.0 ** bfpexp)
    reference = np.fft.fft(inp.astype(np.float64))
    compare(dut, measured, reference, "sine", abs_tol=500)


# ----------------------------------------------------------------------------
# Test 4: two back-to-back frames — pipelining must not corrupt either
# ----------------------------------------------------------------------------
@cocotb.test()
async def test_pipelined_two_frames(dut):
    """Stream two frames back-to-back; verify the overlap is clean.

    Frame 1 is a bin-8 sine, frame 2 is a bin-16 sine with a different
    amplitude (so they have visibly different BFP exponents). The
    handshake stays at one sample per cycle for the full 512 samples;
    backpressure would only kick in if a 3rd frame were streamed before
    compute completed.
    """
    cocotb.start_soon(Clock(dut.i_clk, CLK_PERIOD_NS, units="ns").start())
    await reset_dut(dut)

    n  = np.arange(FFT_SIZE)
    f1 = (5000 * np.sin(2 * np.pi *  8 * n / FFT_SIZE)).astype(np.int32)
    f2 = (3000 * np.sin(2 * np.pi * 16 * n / FFT_SIZE)).astype(np.int32)

    streamer = cocotb.start_soon(stream_samples(dut, np.concatenate([f1, f2])))
    outputs, bfpexps = await capture_output(dut, 2 * FFT_SIZE)
    await streamer

    o1, o2 = outputs[:FFT_SIZE], outputs[FFT_SIZE:]
    e1, e2 = bfpexps[0],         bfpexps[FFT_SIZE]

    dut._log.info(f"[pipelined] frame1 bfpexp={e1}  frame2 bfpexp={e2}")

    measured1 = np.array(o1, dtype=complex) * (2.0 ** e1)
    measured2 = np.array(o2, dtype=complex) * (2.0 ** e2)
    ref1 = np.fft.fft(f1.astype(np.float64))
    ref2 = np.fft.fft(f2.astype(np.float64))

    compare(dut, measured1, ref1, "frame1", abs_tol=500)
    compare(dut, measured2, ref2, "frame2", abs_tol=500)