import os
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles
import numpy as np
from pathlib import Path


# Constants
PCM_RATE  = 16_000
DECIM     = 63
PDM_RATE  = PCM_RATE * DECIM   # 1.008 MHz
FFT_SIZE  = 256
HOP       = FFT_SIZE // 2      # 128  — MUST use // (integer division), not /

# How many PDM samples to generate:
#   We want ~57 output frames from the STFFT with HOP=128.
#   Each frame needs HOP new PCM samples.  With DECIM=63 each PCM sample
#   costs 63 PDM samples.  Add FFT_SIZE worth of PDM samples to fill the
#   first window before any frame fires.
N_PCM_SAMPLES = 7_500
N_PDM_SAMPLES = int((N_PCM_SAMPLES + FFT_SIZE) * DECIM)   # ≈ 476,928

N_MELS = 40
OUT_W  = 16
Q_FRAC = 10

# Extra clocks after the last PDM sample to let the pipeline flush
DRAIN  = 100_000

DATA_DIR = Path(__file__).resolve().parent / ".." / "Log-Mel" / "data"


# Flash helpers

def _idle_flash(dut):
    dut.flash_mel_coeff_we_i.value = 0
    dut.flash_mel_coeff_addr_i.value = 0
    dut.flash_mel_coeff_data_i.value = 0
    dut.flash_mel_index_we_i.value   = 0
    dut.flash_mel_index_addr_i.value = 0
    dut.flash_mel_index_data_i.value = 0
    dut.flash_log_lut_we_i.value     = 0
    dut.flash_log_lut_addr_i.value   = 0
    dut.flash_log_lut_data_i.value   = 0


async def flash_load_all(dut):
    _idle_flash(dut)

    with open(DATA_DIR / "mel_coeffs_sparse.hex") as f:
        coeffs = [int(l.strip(), 16) for l in f if l.strip()]
    cocotb.log.info("Flashing %d sparse mel coeff entries..." % len(coeffs))
    dut.flash_mel_coeff_we_i.value = 1
    for addr, val in enumerate(coeffs):
        dut.flash_mel_coeff_addr_i.value = addr
        dut.flash_mel_coeff_data_i.value = val & 0xFFFF
        await RisingEdge(dut.clk_i)
    dut.flash_mel_coeff_we_i.value = 0
    await ClockCycles(dut.clk_i, 2)

    with open(DATA_DIR / "mel_indices.hex") as f:
        indices = [int(l.strip(), 16) for l in f if l.strip()]
    cocotb.log.info("Flashing %d mel index entries..." % len(indices))
    dut.flash_mel_index_we_i.value = 1
    for addr, val in enumerate(indices):
        dut.flash_mel_index_addr_i.value = addr
        dut.flash_mel_index_data_i.value = val & 0xFF
        await RisingEdge(dut.clk_i)
    dut.flash_mel_index_we_i.value = 0
    await ClockCycles(dut.clk_i, 2)

    with open(DATA_DIR / "log2_lut.hex") as f:
        lut = [int(l.strip(), 16) for l in f if l.strip()]
    cocotb.log.info("Flashing %d log LUT entries..." % len(lut))
    dut.flash_log_lut_we_i.value = 1
    for addr, val in enumerate(lut):
        dut.flash_log_lut_addr_i.value = addr
        dut.flash_log_lut_data_i.value = val & 0xFFFF
        await RisingEdge(dut.clk_i)
    dut.flash_log_lut_we_i.value = 0
    await ClockCycles(dut.clk_i, 2)

    cocotb.log.info("All SRAMs loaded.")



# PCM -> PDM (first-order sigma-delta modulator)

def pcm_to_pdm(pcm: np.ndarray) -> np.ndarray:
    """
    Convert signed 16-bit PCM to 1-bit PDM via first-order sigma-delta.
    Each PCM sample produces exactly DECIM=63 PDM bits.
    Returns int8 array of 0/1 bits, length = len(pcm) * DECIM.
    """
    pdm = []
    acc = 0
    for x in pcm:
        for _ in range(DECIM):
            acc += int(x)
            if acc >= 0:
                pdm.append(1)
                acc -= (1 << 15)
            else:
                pdm.append(0)
                acc += (1 << 15)
    return np.array(pdm, dtype=np.int8)


# Signal generation

def make_chirp(n: int) -> np.ndarray:
    """200 Hz -> 7 kHz linear chirp, signed 16-bit."""
    t   = np.arange(n) / PCM_RATE
    dur = n / PCM_RATE
    phase = 2 * np.pi * (200 * t + (7000 - 200) / (2 * dur) * t**2)
    return (np.sin(phase) * ((1 << 15) - 1)).astype(np.int32)


# Reset

async def do_reset(dut):
    dut.reset_i.value  = 1
    dut.valid_i.value  = 0
    dut.data_i.value   = 0
    dut.test_mode_i.value = 0
    dut.test_coeff_addr_i.value = 0
    dut.test_index_addr_i.value = 0
    dut.test_lut_addr_i.value = 0
    _idle_flash(dut)
    await ClockCycles(dut.clk_i, 20)
    dut.reset_i.value  = 0
    await ClockCycles(dut.clk_i, 10)



# PDM driver

async def drive_pdm(dut, pdm_bits: np.ndarray):
    """
    Drive PDM bits into the DUT one per clock.
    PDM bit 1 -> +32767 (0x7FFF), PDM bit 0 -> -32768 (0x8000).
    """
    for b in pdm_bits:
        dut.data_i.value  = 0x7FFF if b else -0x8000
        dut.valid_i.value = 1
        await RisingEdge(dut.clk_i)

    dut.valid_i.value = 0
    dut.data_i.value  = 0


# Frame collection

async def collect_frames(dut, timeout: int) -> list:
    """
    Collect complete N_MELS-value frames from mel_compensated_o/valid.
    timeout must be an integer number of clock cycles.
    """
    frames = []

    for _ in range(timeout):
        await RisingEdge(dut.clk_i)

        try:
            if int(dut.mel_compensated_valid_o.value):
                v = int(dut.mel_compensated_o.value)

                if not frames or len(frames[-1]) == N_MELS:
                    frames.append([])

                frames[-1].append(v)

        except (ValueError, AttributeError):
            pass

    # Drop incomplete trailing frame
    if frames and len(frames[-1]) < N_MELS:
        frames.pop()

    return frames


# Save features

def save_features(frames: list, name: str = "rtl_features.npy"):
    if not frames:
        cocotb.log.info("No frames to save.")
        return

    mat = np.stack(
        [np.array(f, dtype=np.float32) / (1 << Q_FRAC) for f in frames],
        axis=1   # (N_MELS, n_frames) — matches compare_outputs.py expectation
    )

    path = os.path.join(os.path.dirname(__file__) or ".", name)
    np.save(path, mat)
    cocotb.log.info("Saved %s  shape=%s" % (name, mat.shape))


# Test

@cocotb.test()
async def test_full_pipeline_pdm(dut):
    """
    Full pipeline test:
      PDM input (1.008 MHz) -> CIC -> compFIR -> STFFT -> LogMel -> spect_buffer

    Generates N_PCM_SAMPLES of chirp, converts to PDM, drives the DUT, and
    checks that the expected number of mel-spectrogram frames are produced.
    """
    cocotb.start_soon(Clock(dut.clk_i, 10, units="ns").start())

    await do_reset(dut)
    await flash_load_all(dut)
    await do_reset(dut)

    # Generate audio
    pcm = make_chirp(N_PCM_SAMPLES)
    pdm = pcm_to_pdm(pcm)

    cocotb.log.info(
        "Chirp: %d PCM samples -> %d PDM bits  (DECIM=%d)"
        % (N_PCM_SAMPLES, len(pdm), DECIM)
    )

    # Ideal expected frames: (N_PCM_SAMPLES - FFT_SIZE) // HOP + 1
    # Subtract ~2 for pipeline startup loss.
    ideal    = (N_PCM_SAMPLES - FFT_SIZE) // HOP + 1
    expected_min = max(1, ideal - 5)
    expected_max = ideal

    cocotb.log.info(
        "Expected frames: %d to %d  (ideal=%d, HOP=%d)"
        % (expected_min, expected_max, ideal, HOP)
    )

    timeout = len(pdm) + DRAIN   # integer: len(pdm) is int, DRAIN is int

    cocotb.start_soon(drive_pdm(dut, pdm))
    frames = await collect_frames(dut, timeout)

    n = len(frames)
    cocotb.log.info("Frames produced: %d" % n)

    save_features(frames, "rtl_features.npy")

    # Assertions
    assert n > 0, "No frames produced — check CIC rate, FIR o_tready, STFFT i_ce"

    for i, f in enumerate(frames):
        assert len(f) == N_MELS, "Frame %d has %d mels, expected %d" % (i, len(f), N_MELS)

    assert expected_min <= n <= expected_max, (
        "Frame count %d outside [%d, %d].\n"
        "  If 0 frames: o_tready on compFIR may not be 1, or CIC rate wrong.\n"
        "  If >> expected: fir_trunc_valid stuck high (check FIR handshake)."
        % (n, expected_min, expected_max)
    )

    cocotb.log.info("PASS -- %d frames, %d mels each" % (n, N_MELS))