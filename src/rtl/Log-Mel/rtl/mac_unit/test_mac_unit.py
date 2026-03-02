import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
import numpy as np

POWER_W = 32
COEFF_W = 16
ACCUM_W = 56

# helpers
async def reset_dut(dut):
    dut.reset_i.value      = 1
    dut.power_i.value      = 0
    dut.weight_i.value     = 0
    dut.accumulate_i.value = 0
    dut.clear_i.value      = 0
    await RisingEdge(dut.clk_i)
    await RisingEdge(dut.clk_i)
    dut.reset_i.value = 0

async def accumulate_one(dut, power, weight):
    # one multiply-accumulate operation
    # result appears in accum_o on the next rising edge 
    dut.power_i.value      = power
    dut.weight_i.value     = weight
    dut.accumulate_i.value = 1
    await RisingEdge(dut.clk_i)
    dut.accumulate_i.value = 0

async def clear_mac(dut):
    # reset accumulator
    dut.clear_i.value = 1
    await RisingEdge(dut.clk_i)
    dut.clear_i.value = 0


@cocotb.test()
async def test_mac_single(dut):
    # 10ns period = 100MHz
    cocotb.start_soon(Clock(dut.clk_i, 10, units="ns").start())
    await reset_dut(dut)

    power  = 1000
    weight = 500
    expected = power * weight # no shift

    await accumulate_one(dut, power, weight)
    # need one extra cycle after accumulate_i goes low
    # because accum_o updates on the rising edge after accumulate_i is high
    await RisingEdge(dut.clk_i)

    result = dut.accum_o.value.integer
    assert result == expected, f"Expected {expected} got {result}"
    dut._log.info(f"PASS | power={power} weight={weight} | expected={expected} got={result}")


@cocotb.test()
async def test_mac_accumulate_multiple(dut):
    # multi op accumulate
    cocotb.start_soon(Clock(dut.clk_i, 10, units="ns").start())
    await reset_dut(dut)

    # drive 5 MACs
    cases = [(1000, 500), (2000, 300), (500, 1000), (100, 100), (9999, 1)]
    expected = 0

    for power, weight in cases:
        expected += power * weight
        await accumulate_one(dut, power, weight)

    # wait one extra cycle for last result to register
    await RisingEdge(dut.clk_i)

    result = dut.accum_o.value.integer
    assert result == expected, f"Expected {expected} got {result}"
    dut._log.info(f"PASS | 5 accumulations | expected={expected} got={result}")


@cocotb.test()
async def test_mac_clear(dut):
    # reset test
    cocotb.start_soon(Clock(dut.clk_i, 10, units="ns").start())
    await reset_dut(dut)

    # get a nonzero val first
    await accumulate_one(dut, 1000, 500)
    await RisingEdge(dut.clk_i)

    before_clear = dut.accum_o.value.integer
    assert before_clear != 0, "Accumulator should be non-zero before clear"

    # now clear it
    await clear_mac(dut)
    await RisingEdge(dut.clk_i)

    result = dut.accum_o.value.integer
    assert result == 0, f"Expected 0 after clear, got {result}"
    dut._log.info(f"PASS | clear resets accumulator | before={before_clear} after={result}")


@cocotb.test()
async def test_mac_clear_then_accumulate(dut):
    # accum, clear, then accum again
    cocotb.start_soon(Clock(dut.clk_i, 10, units="ns").start())
    await reset_dut(dut)

    # first frame
    await accumulate_one(dut, 5000, 1000)
    await RisingEdge(dut.clk_i)
    first_result = dut.accum_o.value.integer

    # clear and start second frame
    await clear_mac(dut)

    power  = 200
    weight = 300
    expected = power * weight  # should not include first frame values

    await accumulate_one(dut, power, weight)
    await RisingEdge(dut.clk_i)

    result = dut.accum_o.value.integer
    assert result == expected, f"Expected {expected} got {result} (first frame was {first_result})"
    dut._log.info(f"PASS | clear then new frame | expected={expected} got={result}")


@cocotb.test()
async def test_mac_no_accumulate(dut):
    # dont accumulate when accumulate_i is low
    cocotb.start_soon(Clock(dut.clk_i, 10, units="ns").start())
    await reset_dut(dut)

    # accumulate once to get a non-zero value
    await accumulate_one(dut, 1000, 500)
    await RisingEdge(dut.clk_i)
    snapshot = dut.accum_o.value.integer

    # drive inputs but keep accumulate_i low for 5 cycles
    # accum should not change
    dut.power_i.value  = 9999
    dut.weight_i.value = 9999
    for _ in range(5):
        await RisingEdge(dut.clk_i)

    result = dut.accum_o.value.integer
    assert result == snapshot, f"Accumulator changed without accumulate_i: {snapshot} → {result}"
    dut._log.info(f"PASS | accumulate_i=0 holds value | value={result}")


@cocotb.test()
async def test_mac_golden(dut):
    """Random inputs vs numpy golden model across multiple frames"""
    cocotb.start_soon(Clock(dut.clk_i, 10, units="ns").start())
    await reset_dut(dut)

    rng    = np.random.default_rng(42)
    fails  = 0
    frames = 20  # test 20 simulated frames

    for frame in range(frames):
        await clear_mac(dut)

        # random number of accumulations per frame (1 to 16, like sparse mel filter)
        n_ops    = int(rng.integers(1, 16))
        expected = 0

        for _ in range(n_ops):
            # power is unsigned 31-bit, weight is unsigned 16-bit
            power  = int(rng.integers(0, 2**POWER_W - 1))
            weight = int(rng.integers(0, 2**COEFF_W - 1))
            expected += power * weight
            await accumulate_one(dut, power, weight)

        await RisingEdge(dut.clk_i)
        result = dut.accum_o.value.integer

        if result != expected:
            fails += 1
            dut._log.info(f"FAIL frame={frame} n_ops={n_ops} expected={expected} got={result}")

    if fails == 0:
        dut._log.info(f"PASS | {frames} random frames all correct")
    else:
        raise AssertionError(f"{fails}/{frames} frames failed")
    