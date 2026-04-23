import os
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PCM_RATE   = 16_000
DECIM      = 63
PDM_RATE   = PCM_RATE * DECIM  # 1.008 MHz

N_PCM_SAMPLES = 7500
N_PDM_SAMPLES = N_PCM_SAMPLES * DECIM

N_MELS = 40
OUT_W  = 16
Q_FRAC = 10

DRAIN = 100_000

DATA_DIR = Path(__file__).resolve().parent / ".." / "Log-Mel" / "data"


# ---------------------------------------------------------------------------
# PCM → PDM (Sigma-Delta)
# ---------------------------------------------------------------------------
def pcm_to_pdm(pcm):
    acc = 0
    pdm = []

    for x in pcm:
        acc += int(x)
        if acc >= 0:
            bit = 1
            acc -= (1 << 15)
        else:
            bit = 0
            acc += (1 << 15)
        pdm.append(bit)

    return np.array(pdm, dtype=np.int8)


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------
def make_chirp(n):
    t = np.arange(n) / PCM_RATE
    dur = n / PCM_RATE
    phase = 2 * np.pi * (200 * t + (7000 - 200) / (2 * dur) * t**2)
    return (np.sin(phase) * ((1 << 15) - 1)).astype(np.int32)


# ---------------------------------------------------------------------------
# Flash (reuse your existing logic if you want)
# ---------------------------------------------------------------------------
def _idle_flash(dut):
    dut.flash_mel_coeff_we_i.value   = 0
    dut.flash_mel_index_we_i.value   = 0
    dut.flash_log_lut_we_i.value     = 0


async def do_reset(dut):
    dut.reset_i.value  = 1
    dut.valid_i.value  = 0
    dut.data_i.value   = 0
    _idle_flash(dut)
    await ClockCycles(dut.clk_i, 20)
    dut.reset_i.value  = 0
    await ClockCycles(dut.clk_i, 10)


# ---------------------------------------------------------------------------
# PDM Driver
# ---------------------------------------------------------------------------
async def drive_pdm(dut, pdm_bits):
    for b in pdm_bits:
        if b:
            dut.data_i.value = 0x7FFF
        else:
            dut.data_i.value = -0x8000

        dut.valid_i.value = 1
        await RisingEdge(dut.clk_i)

    dut.valid_i.value = 0
    dut.data_i.value  = 0


# ---------------------------------------------------------------------------
# Frame collection (simplified)
# ---------------------------------------------------------------------------
async def collect_frames(dut, timeout):
    frames = []

    for _ in range(timeout):
        await RisingEdge(dut.clk_i)

        try:
            if int(dut.mel_compensated_valid_o.value):
                v = int(dut.mel_compensated_o.value)

                if not frames or len(frames[-1]) == N_MELS:
                    frames.append([])

                frames[-1].append(v)

        except:
            pass

    if frames and len(frames[-1]) < N_MELS:
        frames.pop()

    return frames

def save_features(frames, name="rtl_features.npy"):
    if not frames:
        return

    mat = np.stack(
        [np.array(f, dtype=np.float32) / (1 << Q_FRAC) for f in frames],
        axis=0   # (frames, mels) ← ML-friendly format
    )

    path = os.path.join(os.path.dirname(__file__) or ".", name)
    np.save(path, mat)

    cocotb.log.info(f"Saved {name} shape={mat.shape}")


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_full_pipeline_pdm(dut):

    cocotb.start_soon(Clock(dut.clk_i, 10, units="ns").start())

    await do_reset(dut)

    # Generate signal
    pcm = make_chirp(N_PCM_SAMPLES)
    pdm = pcm_to_pdm(pcm)

    cocotb.log.info(f"Generated {len(pdm)} PDM samples")

    timeout = N_PDM_SAMPLES + DRAIN

    cocotb.start_soon(drive_pdm(dut, pdm))

    frames = await collect_frames(dut, timeout)

    cocotb.log.info(f"Frames produced: {len(frames)}")

    save_features(frames, "rtl_features.npy")

    assert len(frames) > 0, "No frames produced"

    # Basic sanity
    for i, f in enumerate(frames):
        assert len(f) == N_MELS, f"Frame {i} wrong size"