import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / ".." / ".." / "data"

N_MELS    = 40
ACCUM_W   = 54
LOG_OUT_W = 16
LUT_FRAC  = 6
Q_FRAC    = 10
CLK_PERIOD_NS = 10

ACCUM_MASK = (1 << ACCUM_W) - 1

# Load reference LUT from hex
with open(DATA_DIR / "log2_lut.hex") as _f:
    LOG2_LUT = [int(line.strip(), 16) for line in _f if line.strip()]


# ----------------------------------------------------------------
# Reference model
# ----------------------------------------------------------------

def ref_log_one(energy: int) -> int:
    if energy == 0:
        return 0
    log2_int = int(energy).bit_length() - 1
    MAX_LOG_INT = (1 << (LOG_OUT_W - Q_FRAC)) - 1  # 15
    if log2_int > MAX_LOG_INT:
        return 0xFFFF  # saturation — matches RTL
    mask = (1 << LUT_FRAC) - 1
    if log2_int >= LUT_FRAC:
        addr = (energy >> (log2_int - LUT_FRAC)) & mask
    else:
        addr = (energy << (LUT_FRAC - log2_int)) & mask
    result = (log2_int << Q_FRAC) + LOG2_LUT[addr]
    return result & ((1 << LOG_OUT_W) - 1)


# ----------------------------------------------------------------
# Flash helpers
# ----------------------------------------------------------------

def _idle_flash(dut):
    dut.flash_write_enable_i.value = 0
    dut.flash_addr_i.value = 0
    dut.flash_write_data_i.value = 0


async def flash_load_lut(dut):
    """Flash-load log2_lut.hex into the LUT SRAM (16-bit writes)."""
    _idle_flash(dut)

    with open(DATA_DIR / "log2_lut.hex") as f:
        lut = [int(l.strip(), 16) for l in f if l.strip()]

    cocotb.log.info(f"Flashing {len(lut)} LUT entries...")
    dut.flash_write_enable_i.value = 1
    for addr, val in enumerate(lut):
        dut.flash_addr_i.value = addr
        dut.flash_write_data_i.value = val & 0xFFFF
        await RisingEdge(dut.clk_i)
    dut.flash_write_enable_i.value = 0
    await ClockCycles(dut.clk_i, 2)
    cocotb.log.info("LUT SRAM loaded.")


# ----------------------------------------------------------------
# Stimulus helpers
# ----------------------------------------------------------------

async def reset_dut(dut, cycles: int = 5):
    await RisingEdge(dut.clk_i)
    dut.reset_i.value = 1
    dut.log_en_i.value = 0
    dut.mel_idx_i.value = 0
    _idle_flash(dut)
    # Drive mel_energy_i to zero
    dut.mel_energy_i.value = 0
    await ClockCycles(dut.clk_i, cycles)
    dut.reset_i.value = 0
    await ClockCycles(dut.clk_i, 2)


def pack_mel_energy(energies: np.ndarray) -> int:
    """Pack N_MELS x ACCUM_W-bit values into one big integer for mel_energy_i."""
    val = 0
    for m in range(N_MELS):
        val |= (int(energies[m]) & ACCUM_MASK) << (m * ACCUM_W)
    return val


async def run_log_compress(dut, energies: np.ndarray) -> list:
    """
    Drive mel_energy_i with the given energies, then step mel_idx 0→39
    with log_en=1, mimicking frame_control's LOG_COMPRESS state.

    Returns the 40 log_out values after log_done asserts.
    """
    # Present all mel energies
    dut.mel_energy_i.value = pack_mel_energy(energies)
    await RisingEdge(dut.clk_i)

    # Step through all 40 mel bins with log_en=1
    for idx in range(N_MELS):
        dut.mel_idx_i.value = idx
        dut.log_en_i.value = 1
        await RisingEdge(dut.clk_i)

    dut.log_en_i.value = 0
    dut.mel_idx_i.value = 0

    # Wait for log_done (should come 1-2 cycles after last log_en due to pipeline)
    for i in range(10):
        await RisingEdge(dut.clk_i)
        if dut.log_done_o.value == 1:
            break
    else:
        raise AssertionError("log_done_o never asserted")

    # Read back all 40 log outputs
    raw = int(dut.log_out_o.value)
    mask = (1 << LOG_OUT_W) - 1
    results = [(raw >> (m * LOG_OUT_W)) & mask for m in range(N_MELS)]
    return results


# ----------------------------------------------------------------
# Tests
# ----------------------------------------------------------------

@cocotb.test()
async def test_log_lut_zero(dut):
    """All-zero energy should produce all-zero log output."""
    cocotb.start_soon(Clock(dut.clk_i, CLK_PERIOD_NS, units="ns").start())

    await reset_dut(dut)
    await flash_load_lut(dut)
    await reset_dut(dut)

    energies = np.zeros(N_MELS, dtype=np.uint64)
    got = await run_log_compress(dut, energies)

    cocotb.log.info(f"Zero test got: {got[:5]}...")
    for m in range(N_MELS):
        assert got[m] == 0, f"mel[{m}]: expected 0, got {got[m]}"
    cocotb.log.info("test_log_lut_zero PASSED")


@cocotb.test()
async def test_log_lut_known_values(dut):
    """Test with known energy values and compare to reference model."""
    cocotb.start_soon(Clock(dut.clk_i, CLK_PERIOD_NS, units="ns").start())

    await reset_dut(dut)
    await flash_load_lut(dut)
    await reset_dut(dut)

    # Generate a spread of energy values: powers of 2, random, edge cases
    rng = np.random.default_rng(seed=99)
    energies = np.zeros(N_MELS, dtype=np.uint64)
    energies[0] = 0                        # zero
    energies[1] = 1                        # minimum nonzero
    energies[2] = (1 << ACCUM_W) - 1       # maximum
    energies[3] = 1 << 12                  # power of 2
    energies[4] = 1 << 24                  # larger power of 2
    energies[5] = 1 << 40                  # very large
    energies[6] = 0x123456                 # arbitrary
    energies[7] = 0xDEAD                   # arbitrary
    for m in range(8, N_MELS):
        energies[m] = rng.integers(1, 1 << 48, dtype=np.uint64) & ACCUM_MASK

    got = await run_log_compress(dut, energies)
    exp = [ref_log_one(int(energies[m])) for m in range(N_MELS)]

    n_fail = 0
    for m in range(N_MELS):
        delta = abs(got[m] - exp[m])
        if delta > 1:
            cocotb.log.error(
                f"mel[{m}]: energy=0x{int(energies[m]):x} "
                f"got=0x{got[m]:04x} exp=0x{exp[m]:04x} delta={delta}"
            )
            n_fail += 1
        else:
            cocotb.log.info(
                f"mel[{m}]: energy=0x{int(energies[m]):x} "
                f"got=0x{got[m]:04x} exp=0x{exp[m]:04x} OK"
            )

    assert n_fail == 0, f"{n_fail} mel bins failed"
    cocotb.log.info("test_log_lut_known_values PASSED")


@cocotb.test()
async def test_log_lut_random(dut):
    """Random energies across the full range."""
    cocotb.start_soon(Clock(dut.clk_i, CLK_PERIOD_NS, units="ns").start())

    await reset_dut(dut)
    await flash_load_lut(dut)
    await reset_dut(dut)

    rng = np.random.default_rng(seed=42)
    energies = rng.integers(0, 1 << 48, size=N_MELS, dtype=np.uint64)
    energies = energies & ACCUM_MASK

    got = await run_log_compress(dut, energies)
    exp = [ref_log_one(int(energies[m])) for m in range(N_MELS)]

    deltas = [abs(got[m] - exp[m]) for m in range(N_MELS)]
    worst = max(range(N_MELS), key=lambda m: deltas[m])

    cocotb.log.info(
        f"Random test worst: mel[{worst}] "
        f"got=0x{got[worst]:04x} exp=0x{exp[worst]:04x} delta={deltas[worst]}"
    )
    cocotb.log.info(f"  first 5 got: {[f'0x{g:04x}' for g in got[:5]]}")
    cocotb.log.info(f"  first 5 exp: {[f'0x{e:04x}' for e in exp[:5]]}")

    for m in range(N_MELS):
        assert deltas[m] <= 1, \
            f"mel[{m}]: delta={deltas[m]} (got=0x{got[m]:04x} exp=0x{exp[m]:04x})"

    cocotb.log.info("test_log_lut_random PASSED")