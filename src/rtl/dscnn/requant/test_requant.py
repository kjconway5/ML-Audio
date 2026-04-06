import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, ReadOnly, ClockCycles, Timer

ACC_W  = 32,
DATA_W = 8

@cocotb.test() 
async def positive_nosaturation_test(dut):
    
    dut.relu_en.value = 0
    dut.acc.value = 200
    dut.shift.value = 2
    
    await Timer(1, units='ns')

    assert dut.out.value == 50, f"Expected out to be 50, got {dut.out.value}"
    
    dut.relu_en.value = 0
    dut.acc.value = 100
    dut.shift.value = 2
    
    await Timer(1, units='ns')

    assert dut.out.value == 25, f"Expected out to be 25, got {dut.out.value}"
    
    dut.relu_en.value = 1
    dut.acc.value = 400
    dut.shift.value = 2
    
    await Timer(1, units='ns')

    assert dut.out.value == 100, f"Expected out to be 100, got {dut.out.value}"
    
    dut.relu_en.value = 0
    dut.acc.value = 512
    dut.shift.value = 2
    
    await Timer(1, units='ns')

    assert dut.out.value == 127, f"Expected out to be 127, got {dut.out.value}"
    
    dut.relu_en.value = 0
    dut.acc.value = 0
    dut.shift.value = 2
    
    await Timer(1, units='ns')

    assert dut.out.value == 0, f"Expected out to be 0, got {dut.out.value}"
    
    cocotb.log.info("positive_nosaturation_test passed")
    
@cocotb.test()
async def positive_saturation_test(dut):
    
    dut.relu_en.value = 0
    dut.acc.value = 1025
    dut.shift.value = 2
    
    await Timer(1, units='ns')

    assert dut.out.value == 127, f"Expected out to be 127, got {dut.out.value}"
    
    dut.relu_en.value = 1
    dut.acc.value = 2048
    dut.shift.value = 2
    
    await Timer(1, units='ns')

    assert dut.out.value == 127, f"Expected out to be 127, got {dut.out.value}"
    
    cocotb.log.info("positive_saturation_test passed")
    
