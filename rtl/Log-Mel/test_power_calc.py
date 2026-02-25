import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
import numpy as np

SHIFT = 6
IW    = 18

async def reset_dut(dut):
    dut.valid_il.value = 0
    dut.real_il.value  = 0
    dut.imag_il.value  = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)

def to_signed_bits(val, width):
    if val < 0:
        val = val + (1 << width)
    return val

async def drive_and_read(dut, re, im):
    dut.real_il.value  = to_signed_bits(re, IW)
    dut.imag_il.value  = to_signed_bits(im, IW)
    dut.valid_il.value = 1
    await RisingEdge(dut.clk)
    dut.valid_il.value = 0
    await RisingEdge(dut.clk)
    return dut.power_ol.value.integer

@cocotb.test()
async def test_power_basic(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset_dut(dut)

    re, im   = 100, 200
    expected = (re**2 + im**2) >> SHIFT
    result   = await drive_and_read(dut, re, im)

    assert result == expected, f"Expected {expected}, got {result}"
    dut._log.info(f"PASS | re={re} im={im} | expected={expected} got={result}")


@cocotb.test()
async def test_power_negative_inputs(dut):
    """Make sure squaring changes negative to positive"""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset_dut(dut)

    cases = [(-100, 200), (100, -200), (-100, -200)]
    for re, im in cases:
        expected = (re**2 + im**2) >> SHIFT
        result   = await drive_and_read(dut, re, im)
        assert result == expected, f"re={re} im={im}: expected {expected} got {result}"
        dut._log.info(f"PASS | re={re:+} im={im:+} | expected={expected} got={result}")


@cocotb.test()
async def test_power_golden(dut):
    """200 random inputs vs numpy model"""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset_dut(dut)

    MAX_VAL = 2**(IW-1) - 1
    rng     = np.random.default_rng(42)
    fails   = 0

    for i in range(200):
        re       = int(rng.integers(-MAX_VAL, MAX_VAL))
        im       = int(rng.integers(-MAX_VAL, MAX_VAL))
        expected = (re**2 + im**2) >> SHIFT
        result   = await drive_and_read(dut, re, im)

        if abs(result - expected) > 1:
            fails += 1
            dut._log.info(f"FAIL [{i+1}/200] re={re} im={im} expected={expected} got={result}")

    if fails == 0:
        dut._log.info(f"PASS | 200/200 random tests passed")
    else:
        raise AssertionError(f"{fails}/200 tests failed")