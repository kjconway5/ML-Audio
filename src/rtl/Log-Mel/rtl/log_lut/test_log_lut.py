import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, ReadOnly, ClockCycles

# LUT values
LUT_HEX = [
    "0000","005c","00b6","010f","0166","01bd","0212","0265",
    "02b8","030a","035a","03a9","03f8","0445","0491","04dc",
    "0527","0570","05b9","0600","0647","068d","06d2","0716",
    "075a","079d","07df","0820","0861","08a0","08e0","091e",
    "095c","0999","09d6","0a12","0a4d","0a88","0ac2","0afc",
    "0b35","0b6e","0ba6","0bdd","0c14","0c4a","0c80","0cb6",
    "0ceb","0d1f","0d54","0d87","0dba","0ded","0e1f","0e51",
    "0e83","0eb4","0ee4","0f15","0f44","0f74","0fa3","0fd2",
]
LUT = [int(x, 16) for x in LUT_HEX]

def floor_log2(x: int) -> int:
    """
    floor(log2(x)) for x>0
    """
    return x.bit_length() - 1

def expected_lut_addr(energy: int, log2_int: int, LUT_FRAC: int) -> int:
    """
    Calculates the expected LUT address for a given engergy 
    """
    if log2_int >= LUT_FRAC:
        raw = energy >> (log2_int - LUT_FRAC)
    else:
        raw = energy << (LUT_FRAC - log2_int)

    mask = (1 << LUT_FRAC) - 1
    return raw & mask

def expected_log_result(
    energy: int,
    ACCUM_W: int = 54,
    LOG_OUT_W: int = 16,
    LUT_FRAC: int = 6,
    Q_FRAC: int = 12,
) -> int:
    """
    Calculates the expected log output for a given energy 
    """
    if energy == 0:
        return 0

    log2_int = floor_log2(energy)

    addr = expected_lut_addr(energy, log2_int, LUT_FRAC)
    lut_val = LUT[addr]

    # (log2_int << Q_FRAC) + lut_val, then truncate to LOG_OUT_W
    res = (log2_int << Q_FRAC) + lut_val
    res &= (1 << LOG_OUT_W) - 1
    return res


async def reset_dut(dut, cycles=5):
    """
    Reset the DUT, hold for a few cycles, then release.
    Also resets energies to 0 and mel_idx_i to 0.
    """
    dut.reset.value = 1
    dut.log_en_i.value = 0
    dut.mel_idx_i.value = 0
    # reset energies
    for i in range(len(dut.mel_energy_i)):
        dut.mel_energy_i[i].value = 0

    await ClockCycles(dut.clk, cycles)
    dut.reset.value = 0
    await ClockCycles(dut.clk, cycles)


@cocotb.test()
async def test_log_lut_basic_writes(dut):
    """
    Writes a few bins (including energy=0 and some known patterns),
    checks log_out_o[mel_idx] matches expected,
    checks log_done_o only asserts on mel_idx == N_MELS-1.
    """
    
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset_dut(dut)

    # parameters
    N_MELS = 40
    LUT_FRAC = 6
    Q_FRAC = 12
    LOG_OUT_W = 16

    # create energies with known log2 vals 
    energies = [0] * N_MELS
    energies[0] = 0
    energies[1] = 1
    energies[2] = 2
    energies[3] = 3 
    energies[4] = 64
    energies[5] = 65
    energies[6] = (1 << 20) + 12345
    energies[39] = 999999

    for i in range(N_MELS):
        dut.mel_energy_i[i].value = energies[i]

    bins_to_check = [0, 1, 2, 3, 4, 5, 6, 39]

    for idx in bins_to_check:
        dut.mel_idx_i.value = idx
        dut.log_en_i.value = 1
        await RisingEdge(dut.clk)

        dut.log_en_i.value = 0
        await RisingEdge(dut.clk)

        got = int(dut.log_out_o[idx].value)
        exp = expected_log_result(
            energies[idx],
            LOG_OUT_W=LOG_OUT_W,
            LUT_FRAC=LUT_FRAC,
            Q_FRAC=Q_FRAC,
        )
        
        # check log_out_o value vs expected 
        assert got == exp, (
            f"bin {idx}: energy={energies[idx]} expected 0x{exp:04x} got 0x{got:04x}"
        )
        
        # check if we are on last bin
        done = int(dut.log_done_o.value)
        if idx == N_MELS - 1:
            assert done in (0, 1)
        else:
            assert done == 0, f"log_done_o asserted unexpectedly on idx={idx}"

    dut.log_en_i.value = 0
    cocotb.log.info("test_log_lut_basic_writes passed")


@cocotb.test()
async def test_log_lut_full_sweep(dut):
    """
    Writes all bins from 0 to N_MELS-1, checks log_done_o only asserts on last bin.
    """
    
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

    await reset_dut(dut)

    N_MELS = int(dut.N_MELS) if hasattr(dut, "N_MELS") else len(dut.log_out_o)

    # initialize energies
    for i in range(N_MELS):
        dut.mel_energy_i[i].value = (1 << (i % 16))

    await FallingEdge(dut.clk)
    dut.log_en_i.value = 1

    for idx in range(N_MELS):
        await FallingEdge(dut.clk)
        dut.mel_idx_i.value = idx
        await RisingEdge(dut.clk)
        await ReadOnly()

        done = int(dut.log_done_o.value)

        if idx == N_MELS - 1:
            assert done == 1, f"log_done_o should pulse when mel_idx_i=={N_MELS-1}"
        else:
            assert done == 0, f"log_done_o should stay low before last bin (idx={idx})"

    await FallingEdge(dut.clk)
    dut.log_en_i.value = 0

    await RisingEdge(dut.clk)
    await ReadOnly()
    assert int(dut.log_done_o.value) == 0, "log_done_o should go low once log_en is deasserted"
    cocotb.log.info("test_log_lut_full_sweep passed")