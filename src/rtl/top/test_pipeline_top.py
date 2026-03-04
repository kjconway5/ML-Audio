

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles
import numpy as np

#RTL parameters
SAMPLE_W      = 14
N_MELS        = 40
OUT_W         = 16
CLK_PERIOD_NS = 10
SAMPLE_RATE   = 16000
CE_EVERY      = 20
DRAIN_CLOCKS  = 30_000
SAMPLE_MASK   = (1 << SAMPLE_W) - 1
Q_FRAC        = 12


def make_chirp(n_samples):
    t   = np.arange(n_samples) / SAMPLE_RATE
    dur = n_samples / SAMPLE_RATE
    phase = 2 * np.pi * (200 * t + (7000 - 200) / (2 * dur) * t ** 2)
    peak = (1 << (SAMPLE_W - 1)) - 1
    return (np.sin(phase) * peak).astype(np.int32)


@cocotb.test()
async def test_pipeline(dut):
    #Drive chirp and Verify

    # Clock & reset
    cocotb.start_soon(Clock(dut.clk_i, CLK_PERIOD_NS, unit="ns").start())
    dut.reset_i.value      = 1
    dut.data_i.value       = 0
    dut.valid_i.value      = 0
    dut.cnn_ready_il.value = 1
    await ClockCycles(dut.clk_i, 10)
    dut.reset_i.value = 0
    await ClockCycles(dut.clk_i, 5)

    # Test signal
    samples = make_chirp(7_500)
    n_total = len(samples) * CE_EVERY + DRAIN_CLOCKS
    cocotb.log.info(f"Driving {len(samples)} samples, {n_total} total clocks")

    rtl_frames = []
    sample_idx = 0
    ce_ctr     = 0

    for clk_num in range(n_total):
        await RisingEdge(dut.clk_i)

        # Read CNN output
        try:
            cnn_v = int(dut.cnn_valid_ol.value)
        except ValueError:
            ce_ctr = (ce_ctr + 1) % CE_EVERY
            continue

        if cnn_v:
            try:
                cnn_d = int(dut.cnn_data_ol.value)
            except ValueError:
                cnn_d = 0
            if not rtl_frames or len(rtl_frames[-1]) == N_MELS:
                rtl_frames.append([])
            rtl_frames[-1].append(cnn_d)
            if len(rtl_frames[-1]) == N_MELS:
                cocotb.log.info(f"RTL frame {len(rtl_frames)-1:3d} received")

        
        if clk_num % 20000 == 0:
            cocotb.log.info(
                f"  [clk {clk_num:6d}] sample={sample_idx}/{len(samples)} "
                f"frames={len(rtl_frames)}"
            )

        # Drive next sample
        if ce_ctr == 0 and sample_idx < len(samples):
            dut.data_i.value  = int(samples[sample_idx]) & SAMPLE_MASK
            dut.valid_i.value = 1
            sample_idx += 1
        else:
            dut.valid_i.value = 0

        ce_ctr = (ce_ctr + 1) % CE_EVERY

    # Discard incomplete last frame
    if rtl_frames and len(rtl_frames[-1]) < N_MELS:
        rtl_frames.pop()

    n_rtl = len(rtl_frames)
    cocotb.log.info(f"RTL frames collected: {n_rtl}")

    # ── Assertions
    assert n_rtl > 0, "No CNN output frames collected — check pipeline timing."

    # Check all frames have exactly N_MELS values
    for i, frame in enumerate(rtl_frames):
        assert len(frame) == N_MELS, f"Frame {i} has {len(frame)} bins, expected {N_MELS}"

    # Check values are in valid range [0, 2^OUT_W - 1]
    max_val = (1 << OUT_W) - 1
    for i, frame in enumerate(rtl_frames):
        for j, v in enumerate(frame):
            assert 0 <= v <= max_val, f"Frame {i} mel {j}: value {v} out of range"

    # Check that not all values are zero (signal should produce some energy)
    all_vals = [v for f in rtl_frames for v in f]
    nonzero = sum(1 for v in all_vals if v > 0)
    cocotb.log.info(
        f"Non-zero values: {nonzero}/{len(all_vals)} "
        f"({100*nonzero/len(all_vals):.1f}%)"
    )
    assert nonzero > 0, "All output values are zero — pipeline may not be processing data."

    cocotb.log.info(f"PASS — {n_rtl} frames, all values in valid range")
