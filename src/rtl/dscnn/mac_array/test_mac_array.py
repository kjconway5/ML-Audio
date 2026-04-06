import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, ReadOnly, ClockCycles



async def reset_dut(dut, cycles=5):
    """
    Reset the DUT, hold for a few cycles, then release.
    """
    dut.reset.value = 1
    dut.en.value = 0
    dut.clear.value = 0
    for i in range (8):
        dut.ifmap[i].value = 0
        dut.weight[i].value = 0
    dut.bias.value = 0

    await ClockCycles(dut.clk, cycles)
    dut.reset.value = 0
    await ClockCycles(dut.clk, cycles)

@cocotb.test()
async def reset_test(dut):

    await reset_dut(dut)
    
    assert dut.acc.value == 0, f"Expected acc to be 0 after reset, got {dut.acc.value}"
    assert dut.valid.value == 0, f"Expected valid to be 0"
    
    
    cocotb.log.info("test_log_lut_basic_writes passed")


