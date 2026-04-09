# tb/fft/test_fft_twiddle_rom.py
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge
import numpy as np

def expected_cos_q15(k, N=256):
    """Q1.15 cosine value for twiddle factor k."""
    val = np.cos(2 * np.pi * k / N)
    q15 = int(np.round(val * 32767))
    q15 = max(-32768, min(32767, q15))
    return q15 & 0xFFFF   # as unsigned 16-bit

def q15_to_signed(v):
    return v if v < 32768 else v - 65536

async def read_twiddle(dut, addr):
    await FallingEdge(dut.clk)
    dut.twact.value = 1
    dut.twa.value   = addr
    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)   # registered output: valid 1 cycle later
    dut.twact.value = 0
    return int(dut.twdr_cos.value)

@cocotb.test()
async def test_all_twiddle_values(dut):
    """Verify all 64 twiddle entries match Q1.15 cosine."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await RisingEdge(dut.clk)

    TOLERANCE = 1   # ±1 LSB for rounding
    errors = []

    for k in range(64):
        got      = await read_twiddle(dut, k)
        expected = expected_cos_q15(k)
        got_s    = q15_to_signed(got)
        exp_s    = q15_to_signed(expected)
        if abs(got_s - exp_s) > TOLERANCE:
            errors.append(f"k={k}: expected {exp_s} ({expected:#06x}), "
                          f"got {got_s} ({got:#06x})")

    assert not errors, "Twiddle ROM mismatches:\n" + "\n".join(errors)

@cocotb.test()
async def test_k0_is_one(dut):
    """k=0 must be cos(0)=1.0, which in Q1.15 is 0x7FFF."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await RisingEdge(dut.clk)
    got = await read_twiddle(dut, 0)
    assert got == 0x7FFF, f"k=0 expected 0x7FFF, got {got:#06x}"