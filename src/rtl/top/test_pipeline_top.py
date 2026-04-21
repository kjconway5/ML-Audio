import os
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles
import numpy as np
from pathlib import Path

# Constants
SAMPLE_W     = 14
N_MELS       = 40
OUT_W        = 16
WIN_LEN      = 256
HOP          = 128
CE_EVERY     = 20
DRAIN        = 30_000
N_SAMPLES    = 7_500
STARTUP_LOSS = 3
EXPECTED     = (N_SAMPLES - WIN_LEN) // HOP + 1 - STARTUP_LOSS   # 54
SAMPLE_RATE  = 16_000
Q_FRAC       = 10

DATA_DIR = Path(__file__).resolve().parent / ".." / "Log-Mel" / "data"


# ----------------------------------------------------------------
# Flash helpers
# ----------------------------------------------------------------

def _idle_flash(dut):
    """Drive all flash ports to idle."""
    dut.flash_mel_coeff_we_i.value = 0
    dut.flash_mel_coeff_addr_i.value = 0
    dut.flash_mel_coeff_data_i.value = 0
    dut.flash_mel_index_we_i.value = 0
    dut.flash_mel_index_addr_i.value = 0
    dut.flash_mel_index_data_i.value = 0
    dut.flash_log_lut_we_i.value = 0
    dut.flash_log_lut_addr_i.value = 0
    dut.flash_log_lut_data_i.value = 0


async def flash_load_all(dut):
    """Load all logmel SRAMs via flash ports."""
    _idle_flash(dut)

    # Sparse mel coefficients (16-bit)
    with open(DATA_DIR / "mel_coeffs_sparse.hex") as f:
        coeffs = [int(l.strip(), 16) for l in f if l.strip()]
    cocotb.log.info(f"Flashing {len(coeffs)} sparse mel coeff entries...")
    dut.flash_mel_coeff_we_i.value = 1
    for addr, val in enumerate(coeffs):
        dut.flash_mel_coeff_addr_i.value = addr
        dut.flash_mel_coeff_data_i.value = val & 0xFFFF
        await RisingEdge(dut.clk_i)
    dut.flash_mel_coeff_we_i.value = 0
    await ClockCycles(dut.clk_i, 2)

    # Mel indices (8-bit: starts[0:39], ends[40:79], offsets[80:119])
    with open(DATA_DIR / "mel_indices.hex") as f:
        indices = [int(l.strip(), 16) for l in f if l.strip()]
    cocotb.log.info(f"Flashing {len(indices)} mel index entries...")
    dut.flash_mel_index_we_i.value = 1
    for addr, val in enumerate(indices):
        dut.flash_mel_index_addr_i.value = addr
        dut.flash_mel_index_data_i.value = val & 0xFF
        await RisingEdge(dut.clk_i)
    dut.flash_mel_index_we_i.value = 0
    await ClockCycles(dut.clk_i, 2)

    # Log2 LUT (16-bit, Q6.10)
    with open(DATA_DIR / "log2_lut.hex") as f:
        lut = [int(l.strip(), 16) for l in f if l.strip()]
    cocotb.log.info(f"Flashing {len(lut)} log LUT entries...")
    dut.flash_log_lut_we_i.value = 1
    for addr, val in enumerate(lut):
        dut.flash_log_lut_addr_i.value = addr
        dut.flash_log_lut_data_i.value = val & 0xFFFF
        await RisingEdge(dut.clk_i)
    dut.flash_log_lut_we_i.value = 0
    await ClockCycles(dut.clk_i, 2)

    cocotb.log.info("All SRAMs loaded.")


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def make_chirp(n: int) -> np.ndarray:
    dur = n / SAMPLE_RATE
    t = np.arange(n) / SAMPLE_RATE
    phase = 2 * np.pi * (200 * t + (7000 - 200) / (2 * dur) * t**2)
    return (np.sin(phase) * ((1 << (SAMPLE_W - 1)) - 1)).astype(np.int32)


async def reset(dut):
    dut.reset_i.value = 1
    dut.data_i.value = 0
    dut.valid_i.value = 0
    _idle_flash(dut)
    await ClockCycles(dut.clk_i, 10)
    dut.reset_i.value = 0
    await ClockCycles(dut.clk_i, 5)


async def drive_samples(dut, samples):
    mask = (1 << SAMPLE_W) - 1
    for s in samples:
        dut.data_i.value = int(s) & mask
        dut.valid_i.value = 1
        await RisingEdge(dut.clk_i)
        dut.valid_i.value = 0
        await ClockCycles(dut.clk_i, CE_EVERY - 1)


async def collect_frames(dut, timeout_clks):
    frames: list[list[int]] = []
    for _ in range(timeout_clks):
        await RisingEdge(dut.clk_i)
        try:
            if int(dut.spect_done.value):
                pass
            if int(dut.u_logmel.cnn_valid_ol.value) and int(dut.u_logmel.cnn_ready_il.value):
                v = int(dut.u_logmel.cnn_data_ol.value)
                if not frames or len(frames[-1]) == N_MELS:
                    frames.append([])
                frames[-1].append(v)
        except ValueError:
            pass
    if frames and len(frames[-1]) < N_MELS:
        frames.pop()
    return frames


# ----------------------------------------------------------------
# Test
# ----------------------------------------------------------------

@cocotb.test()
async def test_pipeline(dut):

    cocotb.start_soon(Clock(dut.clk_i, 10, units="ns").start())

    # Initial reset
    await reset(dut)

    # Flash-load all SRAMs
    await flash_load_all(dut)

    # Reset again after flash to ensure clean FSM state
    await reset(dut)

    samples = make_chirp(N_SAMPLES)
    timeout = N_SAMPLES * CE_EVERY + DRAIN

    cocotb.start_soon(drive_samples(dut, samples))
    frames = await collect_frames(dut, timeout)

    n = len(frames)
    cocotb.log.info(f"{N_SAMPLES} samples : {n} frames (expected {EXPECTED})")

    # Frame count
    assert n == EXPECTED, f"frame count: {n} != {EXPECTED}"

    # Each frame has exactly N_MELS values in [0, 2^OUT_W)
    max_val = (1 << OUT_W) - 1
    for i, f in enumerate(frames):
        assert len(f) == N_MELS, f"frame {i}: {len(f)} mels != {N_MELS}"
        for j, v in enumerate(f):
            assert 0 <= v <= max_val, f"frame[{i}][{j}]={v} out of [0,{max_val}]"

    # Pipeline is actually processing (not all zeros)
    all_v = [v for f in frames for v in f]
    nz = sum(1 for v in all_v if v > 0)
    cocotb.log.info(f"non-zero: {nz}/{len(all_v)} ({100*nz/len(all_v):.1f}%)")
    assert nz > 0, "all outputs zero — pipeline not processing"

    # Save for downstream classification
    mat = np.stack(
        [np.array(f, np.float32) / (1 << Q_FRAC) for f in frames], axis=1
    )
    npy = os.path.join(os.path.dirname(__file__) or ".", "rtl_features.npy")
    np.save(npy, mat)
    cocotb.log.info(f"PASS  {n}x{N_MELS} features saved: {npy}")