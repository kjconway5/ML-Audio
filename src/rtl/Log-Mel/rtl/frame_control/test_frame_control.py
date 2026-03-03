import cocotb
from cocotb.triggers import RisingEdge, ClockCycles
from cocotb.clock import Clock

async def reset_dut(dut, cycles=5):
    dut.reset.value = 1
    dut.fft_sync_i.value = 0
    dut.frame_sent_i.value = 0
    dut.filterbank_done_i.value = 0
    await ClockCycles(dut.clk, cycles)
    dut.reset.value = 0
    await ClockCycles(dut.clk, cycles)

def int_sig(x):
    try:
        return int(x.value)
    except Exception:
        return int(x)

@cocotb.test()
async def test_frame_control_fsm(dut):
    """Test frame_control FSM transitions and outputs"""

    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)
    
    MEL_BINS = 40 
    
    # state = IDLE
    await RisingEdge(dut.clk)
    assert int_sig(dut.log_en_o) == 0, "log_en_o should be low in IDLE"
    assert int_sig(dut.output_valid_o) == 0, "output_valid_o should be low in IDLE"

    # IDLE -> ACCUMULATE 
    dut.fft_sync_i.value = 1
    await RisingEdge(dut.clk)
    dut.fft_sync_i.value = 0

    # ACCUMULATE
    await RisingEdge(dut.clk)
    assert int_sig(dut.log_en_o) == 0, "log_en_o should be low in ACCUMULATE"
    assert int_sig(dut.output_valid_o) == 0, "output_valid_o should be low in ACCUMULATE"

    # ACCUMULATE 
    await ClockCycles(dut.clk, 3)
    assert int_sig(dut.log_en_o) == 0, "log_en_o should remain low until LOG_COMPRESS"
    assert int_sig(dut.output_valid_o) == 0, "output_valid_o should remain low until OUTPUT"

    # ACCUMULATE -> LOG_COMPRESS 
    dut.filterbank_done_i.value = 1
    await RisingEdge(dut.clk)
    dut.filterbank_done_i.value = 0

    # LOG_COMPRESS
    await RisingEdge(dut.clk)
    assert int_sig(dut.log_en_o) == 1, "log_en_o should be high in LOG_COMPRESS"
    assert int_sig(dut.output_valid_o) == 0, "output_valid_o should be low in LOG_COMPRESS"

    # LOG_COMPRESS -> OUTPUT
    # have to wait a couple cycles
    reached_output = False
    for _ in range(MEL_BINS + 5):
        await RisingEdge(dut.clk)
        if int_sig(dut.output_valid_o) == 1 and int_sig(dut.log_en_o) == 0:
            reached_output = True
            break

    dut.frame_sent_i.value = 1
    await RisingEdge(dut.clk)
    dut.frame_sent_i.value = 0

    # OUTPUT -> IDLE
    await RisingEdge(dut.clk)
    assert int_sig(dut.output_valid_o) == 0, "output_valid_o should be low back in IDLE"
    assert int_sig(dut.log_en_o) == 0, "log_en_o should be low back in IDLE"
    
    
    
@cocotb.test()
async def test_melindex(dut):
    
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)
    
    # Drive into LOG_COMPRESS
    # IDLE -> ACCUMULATE
    dut.fft_sync_i.value = 1
    await RisingEdge(dut.clk)
    dut.fft_sync_i.value = 0
    await RisingEdge(dut.clk)  # now in ACCUMULATE

    # ACCUMULATE -> LOG_COMPRESS
    dut.filterbank_done_i.value = 1
    await RisingEdge(dut.clk)
    dut.filterbank_done_i.value = 0
    await RisingEdge(dut.clk)  # now in LOG_COMPRESS
    assert int_sig(dut.log_en_o) == 1, "log_en_o should be high in LOG_COMPRESS"

    MEL_BINS = 40  # or: MEL_BINS = int(dut.MEL_BINS) if accessible

    prev = int_sig(dut.mel_idx_o)
    saw = 0

    for _ in range(MEL_BINS + 10):
        await RisingEdge(dut.clk)

        # If we leave LOG_COMPRESS, stop checking index behavior
        if int_sig(dut.log_en_o) == 0:
            break

        cur = int_sig(dut.mel_idx_o)

        # Allow hold or +1 depending on exactly when counter increments, but no big jumps
        assert (cur == prev) or (cur == prev + 1), \
            f"mel_idx_o unexpected change in LOG_COMPRESS: {prev} -> {cur}"

        prev = cur
        saw += 1
        
    assert saw > 0, "Did not see any valid mel_idx_o values in LOG_COMPRESS"
    