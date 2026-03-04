

import os
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles
import numpy as np

# ── Constants 
N_MELS       = 40
OUT_W        = 16
WIN_LEN      = 256
HOP          = 128
CE_EVERY     = 20       # clocks between valid_i pulses (CKPCE=3 + logmel margin)
DRAIN        = 30_000   # flush clocks after last sample
N_SAMPLES    = 7_500
STARTUP_LOSS = 3        # frames consumed by FFT pipeline startup latency
EXPECTED     = (N_SAMPLES - WIN_LEN) // HOP + 1 - STARTUP_LOSS   # 54
SAMPLE_RATE  = 16_000
Q_FRAC       = 12


# Helpers 
def make_chirp(n: int) -> np.ndarray:
    
    dur = n / SAMPLE_RATE
    phase = 2 * np.pi * (200 * t + (7000 - 200) / (2 * dur) * t**2)
    return (np.sin(phase) * ((1 << (SAMPLE_W - 1)) - 1)).astype(np.int32)


async def reset(dut):
    #Assert reset for 10 clocks, deassert, wait 5.
    dut.reset_i.value      = 1
    dut.data_i.value       = 0
    dut.valid_i.value      = 0
    dut.cnn_ready_il.value = 1
    await ClockCycles(dut.clk_i, 10)
    dut.reset_i.value = 0
    await ClockCycles(dut.clk_i, 5)


async def drive_samples(dut, samples):
    #Push each sample for 1 clock, idle for CE_EVERY-1 clocks."""
    mask = (1 << SAMPLE_W) - 1
    for s in samples:
        dut.data_i.value  = int(s) & mask
        dut.valid_i.value = 1
        await RisingEdge(dut.clk_i)
        dut.valid_i.value = 0
        await ClockCycles(dut.clk_i, CE_EVERY - 1)


async def collect_frames(dut, timeout_clks):
    """Collect CNN output until timeout.  Returns list[list[int]]."""
    frames: list[list[int]] = []
    for _ in range(timeout_clks):
        await RisingEdge(dut.clk_i)
        try:
            if int(dut.cnn_valid_ol.value):
                v = int(dut.cnn_data_ol.value)
                if not frames or len(frames[-1]) == N_MELS:
                    frames.append([])
                frames[-1].append(v)
        except ValueError:
            pass
    if frames and len(frames[-1]) < N_MELS:
        frames.pop()
    return frames


# ── Test ────────────────────────────────────────────────────────────────────

@cocotb.test()
async def test_pipeline(dut):
    """Feed chirp → collect log-mel frames → assert count, shape, range."""
    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())
    await reset(dut)

    samples = make_chirp(N_SAMPLES)
    timeout = N_SAMPLES * CE_EVERY + DRAIN

    cocotb.start_soon(drive_samples(dut, samples))
    frames = await collect_frames(dut, timeout)

    n = len(frames)
    cocotb.log.info(f"{N_SAMPLES} samples → {n} frames (expected {EXPECTED})")

    # — frame count
    assert n == EXPECTED, f"frame count: {n} != {EXPECTED}"

    # — each frame has exactly N_MELS values in [0, 2^OUT_W)
    max_val = (1 << OUT_W) - 1
    for i, f in enumerate(frames):
        assert len(f) == N_MELS, f"frame {i}: {len(f)} mels != {N_MELS}"
        for j, v in enumerate(f):
            assert 0 <= v <= max_val, f"frame[{i}][{j}]={v} out of [0,{max_val}]"

    # — pipeline is actually processing (not all zeros)
    all_v = [v for f in frames for v in f]
    nz = sum(1 for v in all_v if v > 0)
    cocotb.log.info(f"non-zero: {nz}/{len(all_v)} ({100*nz/len(all_v):.1f}%)")
    assert nz > 0, "all outputs zero — pipeline not processing"

    # — save for downstream classification (classify_rtl.py / compare_outputs.py)
    mat = np.stack(
        [np.array(f, np.float32) / (1 << Q_FRAC) for f in frames], axis=1
    )
    npy = os.path.join(os.path.dirname(__file__) or ".", "rtl_features.npy")
    np.save(npy, mat)
    cocotb.log.info(f"PASS — {n}×{N_MELS} features saved → {npy}")
