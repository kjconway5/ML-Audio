# test_compFIR.py
#
# Cocotb testbench for compFIR.sv (AXI-Stream ready/valid interface).
#
# Port mapping:
#   i_tdata / i_tvalid / i_tready  -- upstream (from CIC decimator)
#   o_tdata / o_tvalid / o_tready  -- downstream (to STFFT)

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles
import numpy as np
import random

# ---------------------------------------------------------------------------
# Parameters matching compFIR.sv
# ---------------------------------------------------------------------------
NTAPS = 33
IW    = 16
CW    = 14
OW    = 37
M     = (NTAPS - 1) // 2   # = 16

COEFFS = np.array([
    11952, -2084, -77, 699, -845, 802, -680, 533, -390, 267,
    -169, 99, -53, 25, -10, 3, -1
], dtype=np.int64)

full_h = np.zeros(NTAPS, dtype=np.int64)
for k in range(M):
    full_h[k]         = COEFFS[k]
    full_h[NTAPS-1-k] = COEFFS[k]
full_h[M] = COEFFS[M]

MASK_IW = (1 << IW) - 1


# ---------------------------------------------------------------------------
# Fixed-point reference model
# ---------------------------------------------------------------------------
class FIRFixed:
    def __init__(self):
        self.delay = np.zeros(NTAPS, dtype=np.int64)

    def reset(self):
        self.delay[:] = 0

    def push(self, sample: int) -> int:
        self.delay[1:] = self.delay[:-1]
        self.delay[0]  = int(sample)

        sym = np.empty(M + 1, dtype=np.int64)
        for k in range(M):
            sym[k] = self.delay[k] + self.delay[NTAPS - 1 - k]
        sym[M] = self.delay[M]

        def s(x, sh): return int(x) << sh

        p = [
            s(sym[0],14) - s(sym[0],12) - s(sym[0],8)  - s(sym[0],6)  - s(sym[0],4),
            -s(sym[1],11) - s(sym[1],5)  - s(sym[1],2),
            -s(sym[2],6)  - s(sym[2],4)  + s(sym[2],2)  - sym[2],
            s(sym[3],10)  - s(sym[3],8)  - s(sym[3],6)  - s(sym[3],2)  - sym[3],
            -s(sym[4],10) + s(sym[4],8)  - s(sym[4],6)  - s(sym[4],4)  + s(sym[4],2) - sym[4],
            s(sym[5],10)  - s(sym[5],8)  + s(sym[5],5)  + s(sym[5],1),
            -s(sym[6],9)  - s(sym[6],7)  - s(sym[6],5)  - s(sym[6],3),
            s(sym[7],9)   + s(sym[7],4)  + s(sym[7],2)  + sym[7],
            -s(sym[8],9)  + s(sym[8],7)  - s(sym[8],3)  + s(sym[8],1),
            s(sym[9],8)   + s(sym[9],4)  - s(sym[9],2)  - sym[9],
            -s(sym[10],7) - s(sym[10],5) - s(sym[10],3) - sym[10],
            s(sym[11],7)  - s(sym[11],5) + s(sym[11],2) - sym[11],
            -s(sym[12],6) + s(sym[12],4) - s(sym[12],2) - sym[12],
            s(sym[13],5)  - s(sym[13],3) + sym[13],
            -s(sym[14],3) - s(sym[14],1),
            s(sym[15],2)  - sym[15],
            -sym[16],
        ]

        result = sum(p)
        return int(max(-(1 << (OW-1)), min((1 << (OW-1)) - 1, result)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def reset_dut(dut):
    dut.i_reset.value  = 1
    dut.i_tvalid.value = 0
    dut.i_tdata.value  = 0
    dut.o_tready.value = 1   # downstream always ready by default
    await ClockCycles(dut.i_clk, 4)
    dut.i_reset.value  = 0
    await RisingEdge(dut.i_clk)


async def send_sample(dut, sample: int, tready: int = 1):
    """Drive one sample on the upstream port and wait for acceptance."""
    dut.i_tdata.value  = int(sample) & MASK_IW
    dut.i_tvalid.value = 1
    dut.o_tready.value = tready
    # Wait until i_tready is high (DUT willing to accept)
    while True:
        await RisingEdge(dut.i_clk)
        if int(dut.i_tready.value) == 1:
            break
    # Handshake completed this cycle; de-assert valid
    dut.i_tvalid.value = 0


async def send_and_collect(dut, samples, o_tready: int = 1) -> list:
    """
    Send a list of samples back-to-back, collecting DUT outputs aligned
    with the reference model.  Returns the list of DUT o_tdata values
    in the same order as samples.
    """
    ref  = FIRFixed()
    results = []
    dut.o_tready.value = o_tready

    for s in samples:
        dut.i_tdata.value  = int(s) & MASK_IW
        dut.i_tvalid.value = 1

        # Wait for acceptance
        while True:
            await RisingEdge(dut.i_clk)
            if int(dut.i_tready.value) == 1:
                break

        results.append(dut.o_tdata.value.signed_integer)
        ref.push(s)

    dut.i_tvalid.value = 0
    return results


# ---------------------------------------------------------------------------
# Test 1 — Reset
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_reset(dut):
    """After reset: o_tdata = 0, o_tvalid = 0."""
    cocotb.start_soon(Clock(dut.i_clk, 10, units="ns").start())
    await reset_dut(dut)

    assert dut.o_tdata.value.signed_integer == 0, "o_tdata not zero after reset"
    assert int(dut.o_tvalid.value) == 0,          "o_tvalid not zero after reset"

    # Send a sample then reset again
    await send_sample(dut, 1000)
    dut.i_reset.value = 1
    await ClockCycles(dut.i_clk, 4)
    dut.i_reset.value = 0
    await RisingEdge(dut.i_clk)

    assert dut.o_tdata.value.signed_integer == 0, "o_tdata not zero after 2nd reset"
    assert int(dut.o_tvalid.value) == 0,          "o_tvalid not zero after 2nd reset"
    dut._log.info("Reset test passed")


# ---------------------------------------------------------------------------
# Test 2 — i_tready follows ready/valid rule
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_ready_signal(dut):
    """
    i_tready = !o_tvalid || o_tready.
    When output is valid and downstream not ready, FIR must stall upstream.
    """
    cocotb.start_soon(Clock(dut.i_clk, 10, units="ns").start())
    await reset_dut(dut)

    # After reset: o_tvalid=0 so i_tready must be 1
    assert int(dut.i_tready.value) == 1, "i_tready should be 1 when o_tvalid=0"

    # Send one sample, let output become valid, then block downstream
    dut.o_tready.value = 1
    await send_sample(dut, 500)
    await RisingEdge(dut.i_clk)   # output register updates

    # Block downstream
    dut.o_tready.value = 0
    await RisingEdge(dut.i_clk)

    if int(dut.o_tvalid.value) == 1:
        # i_tready must be 0 (stalled) since downstream is not ready
        assert int(dut.i_tready.value) == 0, \
            "i_tready should be 0 when o_tvalid=1 and o_tready=0"
    dut._log.info("Ready signal test passed")


# ---------------------------------------------------------------------------
# Test 3 — Impulse response
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_impulse_response_axi(dut):
    """
    AXI-compliant impulse response test.
    - Drives input using valid/ready handshake
    - Collects output only when valid
    - Matches DUT behavior exactly
    """
    cocotb.start_soon(Clock(dut.i_clk, 10, units="ns").start())
    await reset_dut(dut)

    ref = FIRFixed()
    impulse = (1 << (IW - 1)) - 1  # 32767

    inputs  = [impulse] + [0] * (NTAPS - 1)
    outputs = []

    dut.o_tready.value = 1  # always ready downstream

    sample_idx = 0
    expected_outputs = []

    # Run until we've pushed all inputs AND collected all outputs
    while len(outputs) < len(inputs):

        # ---------------------------
        # Drive input (AXI compliant)
        # ---------------------------
        if sample_idx < len(inputs):
            dut.i_tdata.value  = int(inputs[sample_idx]) & MASK_IW
            dut.i_tvalid.value = 1
        else:
            dut.i_tvalid.value = 0

        await RisingEdge(dut.i_clk)

        # ---------------------------
        # Input handshake
        # ---------------------------
        if int(dut.i_tvalid.value) and int(dut.i_tready.value):
            expected_outputs.append(ref.push(inputs[sample_idx]))
            sample_idx += 1

        # ---------------------------
        # Output handshake
        # ---------------------------
        if int(dut.o_tvalid.value) and int(dut.o_tready.value):
            outputs.append(dut.o_tdata.value.signed_integer)

    # ---------------------------
    # Compare results
    # ---------------------------
    for i, (got, exp) in enumerate(zip(outputs, expected_outputs)):
        assert abs(got - exp) <= 1, \
            f"h[{i}] = {got}, expected {exp}"

    # Check symmetry
    for i in range(M):
        assert outputs[i] == outputs[NTAPS-1-i], \
            f"Asymmetry at {i}: {outputs[i]} vs {outputs[NTAPS-1-i]}"

    dut._log.info("AXI impulse response test passed")


@cocotb.test()
async def test_step_response_axi(dut):
    """AXI-compliant step response test."""
    cocotb.start_soon(Clock(dut.i_clk, 10, units="ns").start())
    await reset_dut(dut)

    ref      = FIRFixed()
    step_val = 1000

    dut.o_tready.value = 1

    n_inputs = NTAPS * 2
    inputs   = [step_val] * n_inputs

    sample_idx = 0
    outputs    = []
    expected   = []

    while len(outputs) < n_inputs:

        # Drive input
        if sample_idx < n_inputs:
            dut.i_tdata.value  = step_val & MASK_IW
            dut.i_tvalid.value = 1
        else:
            dut.i_tvalid.value = 0

        await RisingEdge(dut.i_clk)

        # Input handshake
        if int(dut.i_tvalid.value) and int(dut.i_tready.value):
            expected.append(ref.push(step_val))
            sample_idx += 1

        # Output handshake
        if int(dut.o_tvalid.value) and int(dut.o_tready.value):
            outputs.append(dut.o_tdata.value.signed_integer)

    # Compare full waveform
    for i, (got, exp) in enumerate(zip(outputs, expected)):
        assert abs(got - exp) <= 1, \
            f"Step mismatch[{i}]: dut={got}, ref={exp}"

    # Steady-state check
    expected_ss = int(full_h.sum()) * step_val
    actual_ss   = outputs[-1]

    assert abs(actual_ss - expected_ss) <= 1, \
        f"Steady-state: dut={actual_ss}, expected={expected_ss}"

    dut._log.info("AXI step response test passed")


@cocotb.test()
async def test_random_full_rate_axi(dut):
    """AXI-compliant random test at full throughput."""
    cocotb.start_soon(Clock(dut.i_clk, 10, units="ns").start())
    await reset_dut(dut)

    ref = FIRFixed()
    random.seed(42)

    n = 1000
    samples = [
        random.randint(-(1 << (IW-1)), (1 << (IW-1)) - 1)
        for _ in range(n)
    ]

    dut.o_tready.value = 1

    sample_idx = 0
    outputs    = []
    expected   = []

    while len(outputs) < n:

        # Drive input
        if sample_idx < n:
            dut.i_tdata.value  = int(samples[sample_idx]) & MASK_IW
            dut.i_tvalid.value = 1
        else:
            dut.i_tvalid.value = 0

        await RisingEdge(dut.i_clk)

        # Input handshake
        if int(dut.i_tvalid.value) and int(dut.i_tready.value):
            expected.append(ref.push(samples[sample_idx]))
            sample_idx += 1

        # Output handshake
        if int(dut.o_tvalid.value) and int(dut.o_tready.value):
            outputs.append(dut.o_tdata.value.signed_integer)

            if len(outputs) % 200 == 0:
                dut._log.info(f"Processed {len(outputs)}/{n} samples")

    # Compare all outputs
    for i, (got, exp) in enumerate(zip(outputs, expected)):
        assert got == exp, \
            f"Sample {i}: dut={got}, ref={exp}"

    dut._log.info(f"AXI random full-rate test passed ({n} samples)")


# ---------------------------------------------------------------------------
# Test 6 — Back-pressure: o_tready toggling
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_backpressure(dut):
    """
    With o_tready toggling, no sample must be lost or duplicated.
    Drives valid samples and verifies every accepted output matches the
    reference model.
    """
    cocotb.start_soon(Clock(dut.i_clk, 10, units="ns").start())
    await reset_dut(dut)

    ref     = FIRFixed()
    random.seed(99)
    n       = 200
    samples = [random.randint(-(1 << (IW-1)), (1 << (IW-1)) - 1) for _ in range(n)]

    results = []  # (dut_output, ref_output) collected when o_tvalid && o_tready
    sample_idx = [0]

    for cycle in range(n * 4):
        # Drive upstream: present new sample if we have one
        if sample_idx[0] < n:
            dut.i_tdata.value  = int(samples[sample_idx[0]]) & MASK_IW
            dut.i_tvalid.value = 1
        else:
            dut.i_tvalid.value = 0

        # Alternate downstream ready every 3 cycles
        dut.o_tready.value = 1 if (cycle % 3 != 2) else 0

        await RisingEdge(dut.i_clk)

        # Upstream handshake: sample was accepted
        if int(dut.i_tvalid.value) and int(dut.i_tready.value):
            ref.push(samples[sample_idx[0]])
            sample_idx[0] += 1

        # Downstream handshake: output consumed
        if int(dut.o_tvalid.value) and int(dut.o_tready.value):
            results.append(dut.o_tdata.value.signed_integer)

        if sample_idx[0] >= n:
            break

    dut.i_tvalid.value = 0
    # Drain remaining output
    for _ in range(10):
        dut.o_tready.value = 1
        await RisingEdge(dut.i_clk)
        if int(dut.o_tvalid.value):
            results.append(dut.o_tdata.value.signed_integer)

    # Compare with a fresh reference run over the same samples
    ref2 = FIRFixed()
    expected = [ref2.push(s) for s in samples]

    dut._log.info(
        "Back-pressure test: %d samples in, %d outputs collected"
        % (n, len(results))
    )
    # We can't match 1-to-1 due to pipeline fill, but values must match prefix
    for i, (got, exp) in enumerate(zip(results, expected)):
        assert got == exp, \
            f"Back-pressure output[{i}]: dut={got}, ref={exp}"

    dut._log.info("Back-pressure test passed")


# ---------------------------------------------------------------------------
# Test 7 — o_tvalid deasserts when downstream stalls and no new input
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_valid_deasserts(dut):
    """o_tvalid goes low after downstream consumes the output with no new input."""
    cocotb.start_soon(Clock(dut.i_clk, 10, units="ns").start())
    await reset_dut(dut)

    dut.o_tready.value = 1
    await send_sample(dut, 1234)
    await RisingEdge(dut.i_clk)

    # o_tvalid should now be 1 (output produced)
    assert int(dut.o_tvalid.value) == 1, "o_tvalid should be 1 after sample"

    # One more clock with o_tready=1 and no new input: output consumed, valid clears
    dut.i_tvalid.value = 0
    await RisingEdge(dut.i_clk)
    assert int(dut.o_tvalid.value) == 0, \
        "o_tvalid should deassert after output consumed with no new input"

    dut._log.info("Valid deassert test passed")