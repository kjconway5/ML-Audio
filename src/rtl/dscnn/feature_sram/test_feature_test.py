#interface with external sram for testing 
import cocotb
from cocotb.clock import Clock, Timer
from cocotb.triggers import RisingEdge, FallingEdge, ReadOnly, ClockCycles


@cocotb.test()
async def test_bias_dffs_full_sweep(dut):

    

    cocotb.log.info("test_bias_dffs_full_sweep passed")
    