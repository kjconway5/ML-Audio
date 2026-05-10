import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, ClockCycles, Timer

SPECT_SHIFT = 9
USE_INPUT_REQUANT = 1
INPUT_QUANT_MULT = 5817845
INPUT_QUANT_SHIFT = 31
START_FRAME = 37
N_MELS      = 40
N_FRAMES    = 50
IN_W        = 16
ADDR_W      = 11 
TOTAL_SAMPLES = N_FRAMES * N_MELS
SKIP_SAMPLES = START_FRAME * N_MELS


def expected_quant(q_fixed):
    if USE_INPUT_REQUANT:
        q = (q_fixed * INPUT_QUANT_MULT + (1 << (INPUT_QUANT_SHIFT - 1))) >> INPUT_QUANT_SHIFT
    else:
        q = q_fixed >> SPECT_SHIFT
    return min(q, 127)

async def reset_dut(dut, cycles=5):
    """
    Reset the DUT, hold for a few cycles, then release.
    """
    dut.reset.value = 1
    dut.cnn_valid_i.value = 0

    await ClockCycles(dut.clk, cycles)
    dut.reset.value = 0
    await ClockCycles(dut.clk, cycles)


@cocotb.test() 
async def reset_test(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    await RisingEdge(dut.clk)
    
    await FallingEdge(dut.clk)

    assert dut.wr_addr.value == 0, f"Expected wr_addr to be 0 after reset, got {dut.wr_addr.value}"
    assert dut.spect_done.value == 0, f"Expected spect_done to be 0 after reset, got {dut.spect_done.value}"
    assert dut.spect_write_sel.value == 0, f"Expected spect_write_sel to be 0 after reset, got {dut.spect_write_sel.value}"
    
    cocotb.log.info("reset_test passed")

@cocotb.test()
async def data_out_test(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    await RisingEdge(dut.clk)

    dut.reset.value = 0
    dut.cnn_valid_i.value = 1
    
    # positive within bounds
    dut.cnn_data_i.value = 100
    await Timer(1, unit='ns')
    expected = expected_quant(100)
    assert dut.sp_a_wdata.value == expected, f"Expected sp_a_wdata to be {expected}, got {dut.sp_a_wdata.value}"
    assert dut.sp_b_wdata.value == expected, f"Expected sp_b_wdata to be {expected}, got {dut.sp_b_wdata.value}"

    # positive saturation
    dut.cnn_data_i.value = -1
    await Timer(1, unit='ns')
    assert dut.sp_a_wdata.value == 127, f"Expected sp_a_wdata to be 127, got {dut.sp_a_wdata.value}"
    assert dut.sp_b_wdata.value == 127, f"Expected sp_b_wdata to be 127, got {dut.sp_b_wdata.value}"
    cocotb.log.info("saturation_test passed")
    
    cocotb.log.info("data_out_test passed")
    
@cocotb.test()
async def write_addr_test(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    await RisingEdge(dut.clk)

    dut.reset.value = 0
    dut.cnn_valid_i.value = 1
    dut.cnn_data_i.value = 0
    
    prev_sel = int(dut.spect_write_sel.value)
    await ClockCycles(dut.clk, SKIP_SAMPLES)
    await Timer(1, unit="ns")
    assert dut.wr_addr.value == 0, f"Expected wr_addr to stay 0 before capture, got {dut.wr_addr.value}"

    # After START_FRAME*N_MELS accepted samples, the next edge writes address 0.
    for i in range(TOTAL_SAMPLES):
        await Timer(1, unit="ns")
        assert dut.sp_a_waddr.value == i, f"Expected sp_a_waddr to be {i}, got {dut.sp_a_waddr.value}"
        assert dut.sp_b_waddr.value == i, f"Expected sp_b_waddr to be {i}, got {dut.sp_b_waddr.value}"
        assert dut.sp_a_we.value == (1 if prev_sel == 0 else 0)
        assert dut.sp_b_we.value == (1 if prev_sel == 1 else 0)
        await RisingEdge(dut.clk)

    await Timer(1, unit="ns")
    assert dut.spect_done.value == 1, f"Expected spect_done to be 1 after last sample, got {dut.spect_done.value}"
    assert dut.spect_write_sel.value == 1 - prev_sel, f"Expected spect_write_sel to toggle, got {dut.spect_write_sel.value}"
    assert dut.sp_a_waddr.value == 0, f"Expected sp_a_waddr to wrap to 0, got {dut.sp_a_waddr.value}"
    assert dut.sp_b_waddr.value == 0, f"Expected sp_b_waddr to wrap to 0, got {dut.sp_b_waddr.value}"

    cocotb.log.info("write_addr_test passed")
