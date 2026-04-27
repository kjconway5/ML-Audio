import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles
import numpy as np
import torchaudio.transforms as T
from pathlib import Path
from collections import deque

DATA_DIR = Path(__file__).resolve().parent / "run"

N_MELS, N_BINS, MAX_COEFFS = 40, 129, 16
POWER_W, WEIGHT_W, ACCUM_W = 31, 16, 54
SAMPLE_RATE, N_FFT = 16000, 256
F_MIN, F_MAX = 0.0, SAMPLE_RATE / 2.0

POWER_MAX  = (1 << POWER_W) - 1
WEIGHT_MAX = (1 << WEIGHT_W) - 1
ACCUM_MASK = (1 << ACCUM_W) - 1
CLK_PERIOD_NS = 10
TOLERANCE = 2**26

# Per filter: LOAD_S + LOAD_E + LOAD_O + WAIT_C + WAIT_W + coeffs + DRAIN + LATCH
TIMEOUT_CYCLES = 129 + N_MELS * (MAX_COEFFS + 10) + 50


class My_MelFilterbank:
    """Software reference model — pure numpy, no RTL timing."""
    def __init__(self):
        mel_t = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS,
            f_min=F_MIN, f_max=F_MAX, power=2.0
        )
        fb_fixed = np.round(mel_t.mel_scale.fb.numpy() * (2 ** 15)).astype(np.int64)
        self.fb_fixed = np.clip(fb_fixed, 0, WEIGHT_MAX)

    def compute(self, power_bins: np.ndarray) -> np.ndarray:
        p = (power_bins.astype(np.int64) & POWER_MAX)
        accum = p @ self.fb_fixed
        return (accum & ACCUM_MASK).astype(np.uint64)


class MelScoreboard:
    def __init__(self, ref: My_MelFilterbank):
        self._ref   = ref
        self._queue = deque()
        self.n_checked = 0
        self.n_failed  = 0

    def push(self, power_bins: np.ndarray):
        self._queue.append(self._ref.compute(power_bins))

    def check(self, dut, label: str = ""):
        assert self._queue, "Scoreboard check called but no expected result queued"

        exp    = self._queue.popleft()
        raw    = int(dut.mel_ol.value)
        mask   = (1 << ACCUM_W) - 1
        got    = np.array([(raw >> (m * ACCUM_W)) & mask
                           for m in range(N_MELS)], dtype=np.uint64)
        deltas = np.abs(got.astype(np.int64) - exp.astype(np.int64))
        worst  = int(np.argmax(deltas))

        self.n_checked += 1

        tag = f" [{label}]" if label else ""
        cocotb.log.info(
            f"Scoreboard{tag}: worst mel[{worst}] "
            f"got={got[worst]} exp={exp[worst]} delta={deltas[worst]}"
        )
        cocotb.log.info(f"  first 5 got: {got[:5]}")
        cocotb.log.info(f"  first 5 exp: {exp[:5]}")
        cocotb.log.info(f"  nonzero got={np.count_nonzero(got)} "
                        f"exp={np.count_nonzero(exp)}")

        if not np.all(deltas <= TOLERANCE):
            self.n_failed += 1
            raise AssertionError(
                f"Scoreboard FAIL{tag}: max delta={deltas[worst]} "
                f"at mel[{worst}] (got={got[worst]} exp={exp[worst]})"
            )

        cocotb.log.info(f"Scoreboard{tag}: PASS")

    def summary(self):
        cocotb.log.info(
            f"Scoreboard summary: {self.n_checked} checked, "
            f"{self.n_failed} failed"
        )
        assert self.n_failed == 0, f"{self.n_failed} scoreboard failures"
        assert not self._queue, \
            f"{len(self._queue)} expected results were never checked"


# ----------------------------------------------------------------
# Flash helpers — two SRAMs (index + coeff)
# ----------------------------------------------------------------

def _idle_flash(dut):
    """Drive all flash ports to idle."""
    dut.flash_coeff_we_i.value  = 0
    dut.flash_coeff_addr_i.value = 0
    dut.flash_coeff_data_i.value = 0
    dut.flash_index_we_i.value  = 0
    dut.flash_index_addr_i.value = 0
    dut.flash_index_data_i.value = 0


async def flash_load_all(dut):
    _idle_flash(dut)

    # Load sparse coeff SRAM (16-bit entries)
    with open(DATA_DIR / "mel_coeffs_sparse.hex") as f:
        coeffs = [int(l.strip(), 16) for l in f if l.strip()]
    cocotb.log.info(f"Flashing {len(coeffs)} sparse coeff entries...")
    dut.flash_coeff_we_i.value = 1
    for addr, val in enumerate(coeffs):
        dut.flash_coeff_addr_i.value = addr
        dut.flash_coeff_data_i.value = val & 0xFFFF
        await RisingEdge(dut.clk_i)
    dut.flash_coeff_we_i.value = 0
    await ClockCycles(dut.clk_i, 2)

    # Load index SRAM (8-bit entries: starts[0:39], ends[40:79], offsets[80:119])
    with open(DATA_DIR / "mel_indices.hex") as f:
        indices = [int(l.strip(), 16) for l in f if l.strip()]
    cocotb.log.info(f"Flashing {len(indices)} index entries...")
    dut.flash_index_we_i.value = 1
    for addr, val in enumerate(indices):
        dut.flash_index_addr_i.value = addr
        dut.flash_index_data_i.value = val & 0xFF
        await RisingEdge(dut.clk_i)
    dut.flash_index_we_i.value = 0
    await ClockCycles(dut.clk_i, 2)

    cocotb.log.info("All SRAMs loaded.")


# ----------------------------------------------------------------
# Stimulus helpers
# ----------------------------------------------------------------

async def reset_dut(dut, cycles: int = 5):
    await RisingEdge(dut.clk_i)
    dut.reset_i.value  = 1
    dut.valid_il.value = 0
    dut.test_mode_i.value = 0
    dut.test_coeff_addr_i.value = 0
    dut.test_index_addr_i.value = 0
    _idle_flash(dut)
    await ClockCycles(dut.clk_i, cycles)
    dut.reset_i.value = 0
    await ClockCycles(dut.clk_i, 2)


async def drive_frame(dut, power_bins: np.ndarray):
    for p in power_bins:
        dut.power_il.value = int(p) & POWER_MAX
        dut.valid_il.value = 1
        await RisingEdge(dut.clk_i)
    dut.valid_il.value = 0
    dut.power_il.value = 0


async def wait_for_valid_ol(dut, timeout: int = TIMEOUT_CYCLES) -> int:
    for i in range(timeout):
        await RisingEdge(dut.clk_i)
        if dut.valid_ol.value == 1:
            return i
    raise AssertionError(
        f"Timeout: valid_ol did not assert within {timeout} cycles"
    )


# ----------------------------------------------------------------
# Tests
# ----------------------------------------------------------------

@cocotb.test()
async def test_mel_filterbank_new(dut):
    cocotb.start_soon(Clock(dut.clk_i, CLK_PERIOD_NS, units="ns").start())

    ref = My_MelFilterbank()
    sb  = MelScoreboard(ref)

    await reset_dut(dut)
    await flash_load_all(dut)

    # Reset again after flash to ensure clean state
    await reset_dut(dut)

    # Test 1: Flat
    cocotb.log.info("### TEST 1: Flat ###")
    pb = np.full(N_BINS, 1 << 12, dtype=np.uint64)
    sb.push(pb)
    await drive_frame(dut, pb)
    await wait_for_valid_ol(dut)
    sb.check(dut, label="flat")
    await ClockCycles(dut.clk_i, 1)

    # Test 2: Reset + Random
    cocotb.log.info("### TEST 2: Random & Reset ###")
    await RisingEdge(dut.clk_i)
    dut.reset_i.value  = 1
    dut.valid_il.value = 0
    await ClockCycles(dut.clk_i, 1)
    dut.reset_i.value = 0
    await ClockCycles(dut.clk_i, 2)

    rng = np.random.default_rng(seed=42)
    pb  = rng.integers(0, 1 << 16, size=N_BINS, dtype=np.uint64)
    sb.push(pb)
    await drive_frame(dut, pb)
    await wait_for_valid_ol(dut)
    sb.check(dut, label="random")

    sb.summary()
    cocotb.log.info("ALL TESTS PASSED")