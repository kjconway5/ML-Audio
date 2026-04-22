import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles


async def write_cfg(dut, addr, data):
    dut.cfg_addr.value = addr
    dut.cfg_wdata.value = data
    dut.cfg_we.value = 1
    await RisingEdge(dut.clk)
    dut.cfg_we.value = 0
    await RisingEdge(dut.clk)


async def reset_dut(dut):
    dut.reset.value = 1
    dut.cfg_we.value = 0
    dut.cfg_addr.value = 0
    dut.cfg_wdata.value = 0
    dut.spect_done.value = 0
    dut.spect_write_sel.value = 0
    dut.start.value = 0
    dut.weights_ready.value = 1

    dut.sp_a_rdata.value = 0
    dut.sp_b_rdata.value = 0
    dut.w_data.value = 1
    dut.fs_a_rdata.value = 0
    dut.fs_b_rdata.value = 0
    dut.bias_data.value = 0
    dut.mac_acc.value = 5
    dut.rq_out.value = 7

    await ClockCycles(dut.clk, 5)
    dut.reset.value = 0
    await ClockCycles(dut.clk, 2)


@cocotb.test()
async def test_stays_idle_correctly(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset_dut(dut)

    await ClockCycles(dut.clk, 5)

    assert int(dut.done.value) == 0, "done should stay low in IDLE"
    assert int(dut.state.value) == 0, "FSM should remain in IDLE"


@cocotb.test()
async def test_leaves_idle_correct(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset_dut(dut)

    # ---------- layer 0 tiny config ----------
    # base address for layer 0 = 0x00
    await write_cfg(dut, 0x00, 1)   # cfg_in_ch
    await write_cfg(dut, 0x01, 1)   # cfg_out_ch
    await write_cfg(dut, 0x02, 1)   # cfg_kH
    await write_cfg(dut, 0x03, 1)   # cfg_kW
    await write_cfg(dut, 0x04, 1)   # cfg_stride_h
    await write_cfg(dut, 0x05, 1)   # cfg_stride_w
    await write_cfg(dut, 0x06, 0)   # cfg_pad_h
    await write_cfg(dut, 0x07, 0)   # cfg_pad_w
    await write_cfg(dut, 0x08, 0)   # cfg_dw
    await write_cfg(dut, 0x09, 0)   # cfg_w_off[7:0]
    await write_cfg(dut, 0x0A, 0)   # cfg_w_off[12:8]
    await write_cfg(dut, 0x0B, 0)   # cfg_shift
    await write_cfg(dut, 0x0C, 0)   # cfg_relu
    await write_cfg(dut, 0x0D, 1)   # cfg_ofmap_h
    await write_cfg(dut, 0x0E, 1)   # cfg_ofmap_w
    await write_cfg(dut, 0x0F, 0)   # cfg_bias_off

    # mark config done
    await write_cfg(dut, 0xFF, 1)

    # make spectrogram ready
    dut.spect_write_sel.value = 0
    dut.spect_done.value = 1
    await RisingEdge(dut.clk)
    dut.spect_done.value = 0
    await RisingEdge(dut.clk)

    # start pulse
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    await RisingEdge(dut.clk)

    assert int(dut.state.value) != 0, "FSM should leave IDLE after start/cfg/spect_ready"


@cocotb.test()
async def test_fsm_single_layer_reaches_done(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset_dut(dut)

    # -------------------------
    # Configure all 10 layers
    # Keep everything tiny so FSM finishes quickly
    # layer 0 uses spectrogram input
    # later layers use feature SRAM
    # -------------------------
    for layer in range(10):
        base = layer << 4

        await write_cfg(dut, base + 0, 1)   # cfg_in_ch
        await write_cfg(dut, base + 1, 1)   # cfg_out_ch
        await write_cfg(dut, base + 2, 1)   # cfg_kH
        await write_cfg(dut, base + 3, 1)   # cfg_kW
        await write_cfg(dut, base + 4, 1)   # cfg_stride_h
        await write_cfg(dut, base + 5, 1)   # cfg_stride_w
        await write_cfg(dut, base + 6, 0)   # cfg_pad_h
        await write_cfg(dut, base + 7, 0)   # cfg_pad_w
        await write_cfg(dut, base + 8, 0)   # cfg_dw = pointwise
        await write_cfg(dut, base + 9, 0)   # cfg_w_off low
        await write_cfg(dut, base + 10, 0)  # cfg_w_off high
        await write_cfg(dut, base + 11, 0)  # cfg_shift
        await write_cfg(dut, base + 12, 0)  # cfg_relu
        await write_cfg(dut, base + 13, 1)  # cfg_ofmap_h = 1
        await write_cfg(dut, base + 14, 1)  # cfg_ofmap_w = 1
        await write_cfg(dut, base + 15, 0)  # cfg_bias_off

    # config complete
    await write_cfg(dut, 0xFF, 1)

    # spectrogram available
    dut.spect_write_sel.value = 0
    dut.spect_done.value = 1
    await RisingEdge(dut.clk)
    dut.spect_done.value = 0
    await RisingEdge(dut.clk)

    # input data / external blocks
    dut.sp_a_rdata.value = 3
    dut.sp_b_rdata.value = 4
    dut.fs_a_rdata.value = 2
    dut.fs_b_rdata.value = 2
    dut.w_data.value = 1
    dut.mac_acc.value = 9
    dut.rq_out.value = 6

    # start
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # wait for completion
    done_seen = False
    for _ in range(500):
        await RisingEdge(dut.clk)
        if int(dut.done.value) == 1:
            done_seen = True
            break

    assert done_seen, "FSM did not assert done"
    assert int(dut.class_out.value) == 0, "With 1 output channel, class_out should be 0"