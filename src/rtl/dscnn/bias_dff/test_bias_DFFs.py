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
DEPTH = 295
DATA_W = 32


@cocotb.test()
async def test_bias_dffs_basic_addresses(dut):
    test_addrs = [
        0, 1, 31, 32, 63, 64, 95, 96,
        127, 128, 159, 160, 191, 192,
        223, 255, 256, 287, 288, 294,
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
    for addr in range(295):
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
        0, 31,       # first_conv
        32, 63,      # ds_blocks.0.depthwise
        64, 95,      # ds_blocks.0.pointwise
        96, 127,     # ds_blocks.1.depthwise
        128, 159,    # ds_blocks.1.pointwise
        160, 191,    # ds_blocks.2.depthwise
        192, 223,    # ds_blocks.2.pointwise
        224, 255,    # ds_blocks.3.depthwise
        256, 287,    # ds_blocks.3.pointwise
        288, 294,    # classifier
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