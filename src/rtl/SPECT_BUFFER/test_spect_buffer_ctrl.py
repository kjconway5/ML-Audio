import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, ReadOnly, ClockCycles, Timer

SPECT_SHIFT = 4
N_MELS      = 40
N_FRAMES    = 50
IN_W        = 16
ADDR_W      = 11 
TOTAL_SAMPLES = N_FRAMES * N_MELS;

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

@cocotb.test()
async def data_out_test(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset_dut(dut)
    await RisingEdge(dut.clk)

    dut.reset.value = 0
    dut.cnn_valid_i.value = 1
    
    # positive within bounds
    dut.cnn_data_i.value = 100
    await Timer(1, units='ns')
    assert dut.sp_a_wdata.value == 6, f"Expected sp_a_wdata to be 6, got {dut.sp_a_wdata.value}"
    assert dut.sp_b_wdata.value == 6, f"Expected sp_b_wdata to be 6, got {dut.sp_b_wdata.value}"

    # negative within bounds
    # output is large positive, but the rtl knows its signed and 
    # in twos complement
    dut.cnn_data_i.value = -100
    await Timer(1, units='ns')
    assert dut.sp_a_wdata.value == 249, f"Expected sp_a_wdata to be 249, got {dut.sp_a_wdata.value}"
    assert dut.sp_b_wdata.value == 249, f"Expected sp_b_wdata to be 249, got {dut.sp_b_wdata.value}"

    # positive saturation
    dut.cnn_data_i.value = 4092
    await Timer(1, units='ns')
    assert dut.sp_a_wdata.value == 127, f"Expected sp_a_wdata to be 127, got {dut.sp_a_wdata.value}"
    assert dut.sp_b_wdata.value == 127, f"Expected sp_b_wdata to be 127, got {dut.sp_b_wdata.value}"

    # negative saturation
    # output is large positive, but the rtl knows its signed and 
    # in twos complement
    dut.cnn_data_i.value = -4092
    await Timer(1, units='ns')
    assert dut.sp_a_wdata.value == 128, f"Expected sp_a_wdata to be 128, got {dut.sp_a_wdata.value}"
    assert dut.sp_b_wdata.value == 128, f"Expected sp_b_wdata to be 128, got {dut.sp_b_wdata.value}"
    cocotb.log.info("saturation_test passed")
    
    cocotb.log.info("data_out_test passed")
    
@cocotb.test()
async def write_addr_test(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset_dut(dut)
    await RisingEdge(dut.clk)

    dut.reset.value = 0
    dut.cnn_valid_i.value = 1
    if dut.spect_write_sel.value == 0:
        prev_sel = 0
    else:
        prev_sel = 1
    for i in range(TOTAL_SAMPLES):
        dut.wr_addr.value = i
        await Timer(1, units='ns')
        if i < TOTAL_SAMPLES:
            last_addr = i
            assert dut.sp_a_waddr.value == last_addr, f"Expected sp_a_waddr to be {last_addr}, got {dut.sp_a_waddr.value}"
            assert dut.sp_b_waddr.value == last_addr, f"Expected sp_b_waddr to be {last_addr}, got {dut.sp_b_waddr.value}"
        else:
            assert dut.sp_a_waddr.value == 0, f"Expected sp_a_waddr to wrap to 0, got {dut.sp_a_waddr.value}"
            assert dut.sp_b_waddr.value == 0, f"Expected sp_b_waddr to wrap to 0, got {dut.sp_b_waddr.value}"
            assert dut.spect_done.value == 1, f"Expected spect_done to be 1 after last sample, got {dut.spect_done.value}"
            assert dut.spect_write_sel.value == prev_sel, f"Expected spect_write_sel to be {prev_sel}, got {dut.spect_write_sel.value}"
    
    
    
    cocotb.log.info("write_addr_test passed")
    