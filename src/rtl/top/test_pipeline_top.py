import cocotb
from cocotb.triggers import RisingEdge, ClockCycles
from cocotb.clock import Clock

async def reset_dut(dut, cycles=5):
    """
    Reset the DUT, hold for a few cycles, then release.
    Also resets control inputs to 0
    """
    dut.reset.value = 1
    dut.fft_sync_i.value = 0
    dut.frame_sent_i.value = 0
    dut.filterbank_done_i.value = 0
    await ClockCycles(dut.clk, cycles)
    dut.reset.value = 0
    await ClockCycles(dut.clk, cycles)


@cocotb.test()
async def test_pipeline_top(dut):
    """
    Test pipeline_top
    """

    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)
    
    