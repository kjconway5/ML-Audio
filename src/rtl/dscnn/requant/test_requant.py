import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, ReadOnly, ClockCycles, Timer

ACC_W  = 32,
DATA_W = 8

@cocotb.test()
async def positive_nosaturation_test(dut):
    # mult=0x40000000 (2^30) with shift=0 gives effective scale of 1/4,
    # matching the old shift=2 behavior: out = (acc * 2^30) >> 32 >> 0 = acc / 4
    dut.relu_en.value = 0
    dut.acc.value = 200
    dut.mult.value = 0x40000000
    dut.shift.value = 0

    await Timer(1, units='ns')

    assert dut.out.value == 50, f"Expected out to be 50, got {dut.out.value}"

    dut.relu_en.value = 0
    dut.acc.value = 100
    dut.mult.value = 0x40000000
    dut.shift.value = 0

    await Timer(1, units='ns')

    assert dut.out.value == 25, f"Expected out to be 25, got {dut.out.value}"

    dut.relu_en.value = 1
    dut.acc.value = 400
    dut.mult.value = 0x40000000
    dut.shift.value = 0

    await Timer(1, units='ns')

    assert dut.out.value == 100, f"Expected out to be 100, got {dut.out.value}"

    dut.relu_en.value = 0
    dut.acc.value = 512
    dut.mult.value = 0x40000000
    dut.shift.value = 0

    await Timer(1, units='ns')

    assert dut.out.value == 127, f"Expected out to be 127, got {dut.out.value}"

    dut.relu_en.value = 0
    dut.acc.value = 0
    dut.mult.value = 0x40000000
    dut.shift.value = 0

    await Timer(1, units='ns')

    assert dut.out.value == 0, f"Expected out to be 0, got {dut.out.value}"

    cocotb.log.info("positive_nosaturation_test passed")
    
@cocotb.test()
async def positive_saturation_test(dut):
    dut.relu_en.value = 0
    dut.acc.value = 1025
    dut.mult.value = 0x40000000
    dut.shift.value = 0

    await Timer(1, units='ns')

    assert dut.out.value == 127, f"Expected out to be 127, got {dut.out.value}"

    dut.relu_en.value = 1
    dut.acc.value = 2048
    dut.mult.value = 0x40000000
    dut.shift.value = 0

    await Timer(1, units='ns')

    assert dut.out.value == 127, f"Expected out to be 127, got {dut.out.value}"

    cocotb.log.info("positive_saturation_test passed")
    
@cocotb.test()
async def negative_test(dut):
    dut.relu_en.value = 0
    dut.acc.value = -200
    dut.mult.value = 0x40000000
    dut.shift.value = 0

    await Timer(1, units='ns')

    assert dut.out.value == 206, f"Expected out to be 206, got {dut.out.value}"

    dut.relu_en.value = 0
    dut.acc.value = -100
    dut.mult.value = 0x40000000
    dut.shift.value = 0

    await Timer(1, units='ns')

    assert dut.out.value == 231, f"Expected out to be 231, got {dut.out.value}"

    dut.relu_en.value = 1
    dut.acc.value = -400
    dut.mult.value = 0x40000000
    dut.shift.value = 0

    await Timer(1, units='ns')

    assert dut.out.value == 0, f"Expected out to be 0, got {dut.out.value}"

    cocotb.log.info("negative_test passed")

@cocotb.test()
async def negative_saturation_test(dut):
    dut.relu_en.value = 0
    dut.acc.value = -1025
    dut.mult.value = 0x40000000
    dut.shift.value = 0

    await Timer(1, units='ns')

    assert dut.out.value == 128, f"Expected out to be 128, got {dut.out.value}"

    dut.relu_en.value = 1
    dut.acc.value = -2048
    dut.mult.value = 0x40000000
    dut.shift.value = 0

    await Timer(1, units='ns')

    assert dut.out.value == 0, f"Expected out to be 0, got {dut.out.value}"

    cocotb.log.info("negative_saturation_test passed")
    
@cocotb.test()
async def relu_test(dut):
    dut.relu_en.value = 1
    dut.acc.value = -200
    dut.shift.value = 2
    
    await Timer(1, units='ns')

    assert dut.out.value == 0, f"Expected out to be 0, got {dut.out.value}"
    
    dut.relu_en.value = 1
    dut.acc.value = -100
    dut.shift.value = 2
    
    await Timer(1, units='ns')

    assert dut.out.value == 0, f"Expected out to be 0, got {dut.out.value}"
    
    dut.relu_en.value = 1
    dut.acc.value = -400
    dut.shift.value = 2
    
    await Timer(1, units='ns')

    assert dut.out.value == 0, f"Expected out to be 0, got {dut.out.value}"
    
    cocotb.log.info("relu_test passed")