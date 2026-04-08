import cocotb
from cocotb.clock import Clock, Timer
from cocotb.triggers import RisingEdge, FallingEdge, ReadOnly, ClockCycles
from pathlib import Path


def load_hex_file(filename):
    hex_path = Path(__file__).parent / filename

    values = []
    with open(hex_path, "r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("//"):
                continue

            raw = int(line, 16)

            # Convert unsigned hex to signed 32-bit integer
            if raw & (1 << 31):
                raw -= (1 << 32)

            values.append(raw)

    return values


EXPECTED_HEX = load_hex_file("bias.hex")
DEPTH = 223
DATA_W = 32


@cocotb.test()
async def test_bias_dffs_basic_addresses(dut):
    test_addrs = [
        0, 1, 23, 24, 48, 71, 72, 96,
        120, 144, 168, 192, 216, 222,
        223, 255,
    ]

    for addr in test_addrs:
        dut.addr.value = addr
        await Timer(1, units="ns")

        got = dut.data.value.signed_integer
        exp = EXPECTED_HEX[addr] if addr < len(EXPECTED_HEX) else 0

        assert got == exp, (
            f"addr={addr}: expected {exp} (0x{(exp & 0xFFFFFFFF):08X}), "
            f"got {got} (0x{(got & 0xFFFFFFFF):08X})"
        )

    cocotb.log.info("test_bias_dffs_basic_addresses passed")


@cocotb.test()
async def test_bias_dffs_full_sweep(dut):
    for addr in range(256):
        dut.addr.value = addr
        await Timer(1, units="ns")

        got = dut.data.value.signed_integer
        exp = EXPECTED_HEX[addr] if addr < len(EXPECTED_HEX) else 0

        assert got == exp, (
            f"addr={addr}: expected {exp} (0x{(exp & 0xFFFFFFFF):08X}), "
            f"got {got} (0x{(got & 0xFFFFFFFF):08X})"
        )

    cocotb.log.info("test_bias_dffs_full_sweep passed")


@cocotb.test()
async def test_bias_dffs_block_boundaries(dut):
    boundary_addrs = [
        0, 23,
        24, 47,
        48, 71,
        72, 95,
        96, 119,
        120, 143,
        144, 167,
        168, 191,
        192, 215,
        216, 222,
    ]

    for addr in boundary_addrs:
        dut.addr.value = addr
        await Timer(1, units="ns")

        got = dut.data.value.signed_integer
        exp = EXPECTED_HEX[addr] if addr < len(EXPECTED_HEX) else 0

        assert got == exp, (
            f"addr={addr}: expected {exp} (0x{(exp & 0xFFFFFFFF):08X}), "
            f"got {got} (0x{(got & 0xFFFFFFFF):08X})"
        )

    cocotb.log.info("test_bias_dffs_block_boundaries passed")