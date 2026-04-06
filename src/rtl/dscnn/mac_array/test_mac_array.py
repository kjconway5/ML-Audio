import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, ReadOnly, ClockCycles

N_MACS = 16
DATA_W = 8
ACC_W  = 32

async def reset_dut(dut, cycles=5):
    """
    Reset the DUT, hold for a few cycles, then release.
    """
    dut.reset.value = 1
    dut.en.value = 0
    dut.clear.value = 0
    for i in range (N_MACS):
        dut.ifmap[i].value = 0
        dut.weight[i].value = 0
    dut.bias.value = 0

    await ClockCycles(dut.clk, cycles)
    dut.reset.value = 0
    await ClockCycles(dut.clk, cycles)

@cocotb.test()
async def reset_test(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset_dut(dut)
    await RisingEdge(dut.clk)
    
    await FallingEdge(dut.clk)
    
    assert dut.acc.value == 0, f"Expected acc to be 0 after reset, got {dut.acc.value}"
    assert dut.valid.value == 0, f"Expected valid to be 0"
    
    
    cocotb.log.info("reset_test passed")


@cocotb.test() 
async def clear_test(dut):
    cocotb.start_soon(Clock(dut.clk, 20, units="ns").start())
    await reset_dut(dut)
    await RisingEdge(dut.clk)

    # set some non-zero values
    dut.clear.value = 1
    dut.reset.value = 0
    dut.en.value = 0
    
    for i in range (N_MACS):
        dut.ifmap[i].value = i + 1
        dut.weight[i].value = (i + 1) * 2
    dut.bias.value = 123
    

    assert dut.acc.value == dut.bias.value, f"Expected acc to be bias after clear, got {dut.acc.value}"
    assert dut.valid.value == 0, f"Expected valid to be 0 after clear"
    
    cocotb.log.info("clear_test passed")
    
@cocotb.test()
async def en_test(dut):
    cocotb.start_soon(Clock(dut.clk, 20, units="ns").start())
    await reset_dut(dut)
    await RisingEdge(dut.clk)
    
    dut.en.value = 1
    dut.clear.value = 0
    dut.reset.value = 0
    sum = 0
    
    # set some non-zero values
    for i in range (N_MACS):
        dut.ifmap[i].value = i * 3 + 1
        dut.weight[i].value = (i + 1) * 2
        sum = sum + (dut.ifmap[i].value * dut.weight[i].value)
        
    await FallingEdge(dut.clk)
    assert dut.acc.value ==  sum, f"Expected acc to be the sum of all MAC operations, got {dut.acc.value}"
    assert dut.valid.value == 1, f"Expected valid to be 1 after en, got {dut.valid.value}"
    
    cocotb.log.info("en_test passed")
    
@cocotb.test() 
async def else_test(dut):
    cocotb.start_soon(Clock(dut.clk, 20, units="ns").start())
    await reset_dut(dut)
    await RisingEdge(dut.clk)
    
    # set some non-zero values
    dut.en.value = 0
    dut.reset.value = 0
    dut.clear.value = 0
        
    assert dut.valid.value == 0, f"Expected valid to be 0 when en=0, got {dut.valid.value}"
    
    cocotb.log.info("else_test passed")