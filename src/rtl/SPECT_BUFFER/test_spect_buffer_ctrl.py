import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, ReadOnly, ClockCycles, Timer

SPECT_SHIFT = 4
N_MELS      = 40
N_FRAMES    = 50
IN_W        = 16
ADDR_W      = 11 

async def reset_dut(dut, cycles=5):
    """
    Reset the DUT, hold for a few cycles, then release.
    """
    dut.reset.value = 1
    dut.cnn_valid_i.value = 0
    dut.cnn_ready_o.value = 0

    await ClockCycles(dut.clk, cycles)
    dut.reset.value = 0
    await ClockCycles(dut.clk, cycles)


@cocotb.test() 
async def reset_test(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset_dut(dut)
    await RisingEdge(dut.clk)
    
    await FallingEdge(dut.clk)

    assert dut.wr_addr.value == 0, f"Expected wr_addr to be 0 after reset, got {dut.wr_addr.value}"
    assert dut.spect_done.value == 0, f"Expected spect_done to be 0 after reset, got {dut.spect_done.value}"
    assert dut.spect_write_sel.value == 0, f"Expected spect_write_sel to be 0 after reset, got {dut.spect_write_sel.value}"
    
    cocotb.log.info("reset_test passed")

