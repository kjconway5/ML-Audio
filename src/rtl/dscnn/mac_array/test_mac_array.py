import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, ClockCycles

DATA_W = 8
ACC_W  = 32

async def reset_dut(dut, cycles=5):
    dut.reset.value = 1
    dut.en.value = 0
    dut.clear.value = 0
    dut.ifmap.value = 0
    dut.weight.value = 0
    dut.bias.value = 0

    await ClockCycles(dut.clk, cycles)
    dut.reset.value = 0
    await ClockCycles(dut.clk, cycles)

@cocotb.test()
async def reset_test(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset_dut(dut)
    await FallingEdge(dut.clk)

    assert dut.acc.value == 0, f"Expected acc=0 after reset, got {dut.acc.value}"
    cocotb.log.info("reset_test passed")


@cocotb.test()
async def clear_test(dut):
    cocotb.start_soon(Clock(dut.clk, 20, units="ns").start())
    await reset_dut(dut)

    dut.clear.value = 1
    dut.reset.value = 0
    dut.en.value = 0
    dut.ifmap.value = 5
    dut.weight.value = 10
    dut.bias.value = 123

    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)

    assert dut.acc.value == 123, f"Expected acc=bias(123) after clear, got {dut.acc.value}"
    cocotb.log.info("clear_test passed")


@cocotb.test()
async def en_test(dut):
    cocotb.start_soon(Clock(dut.clk, 20, units="ns").start())
    await reset_dut(dut)

    dut.en.value = 1
    dut.clear.value = 0
    dut.reset.value = 0
    dut.ifmap.value = 3
    dut.weight.value = 5

    # First accumulation: acc = 0 + 3*5 = 15
    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    assert dut.acc.value == 15, f"Expected acc=15, got {dut.acc.value}"

    # Second accumulation: acc = 15 + 3*5 = 30
    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    assert dut.acc.value == 30, f"Expected acc=30 after second accumulation, got {dut.acc.value}"

    cocotb.log.info("en_test passed")


@cocotb.test()
async def else_test(dut):
    cocotb.start_soon(Clock(dut.clk, 20, units="ns").start())
    await reset_dut(dut)

    # With en=0, acc should hold its value (0 after reset)
    dut.en.value = 0
    dut.reset.value = 0
    dut.clear.value = 0
    dut.ifmap.value = 99
    dut.weight.value = 99

    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)

    assert dut.acc.value == 0, f"Expected acc to hold 0 when en=0, got {dut.acc.value}"
    cocotb.log.info("else_test passed")
