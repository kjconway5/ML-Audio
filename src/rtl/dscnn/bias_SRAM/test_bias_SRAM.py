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
DEPTH = 151
DATA_W = 32


async def init_dut(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    dut.we.value = 0
    dut.waddr.value = 0
    dut.wdata.value = 0
    dut.addr.value = 0
    dut.read_high.value = 0
    await ClockCycles(dut.clk, 2)


async def read_bias(dut, addr):
    # 2-cycle read: present addr with read_high=0, then read_high=1.
    # SIM model pipelines addr through raddr_q0→raddr_q1; data valid after 2 edges.
    await FallingEdge(dut.clk)
    dut.addr.value = addr
    dut.read_high.value = 0
    await RisingEdge(dut.clk)   # edge 1: raddr_q0 <= addr
    await FallingEdge(dut.clk)
    dut.read_high.value = 1
    await RisingEdge(dut.clk)   # edge 2: raddr_q1 <= raddr_q0; data valid
    await ReadOnly()
    return dut.data.value.signed_integer


@cocotb.test()
async def test_bias_sram_basic_addresses(dut):
    await init_dut(dut)
    test_addrs = [
        0, 1, 15, 16, 31, 32, 47, 48,
        63, 64, 79, 80, 95, 112, 128, 143, 150,
    ]

    for addr in test_addrs:
        got = await read_bias(dut, addr)
        exp = EXPECTED_HEX[addr] if addr < len(EXPECTED_HEX) else 0

        assert got == exp, (
            f"addr={addr}: expected {exp} (0x{(exp & 0xFFFFFFFF):08X}), "
            f"got {got} (0x{(got & 0xFFFFFFFF):08X})"
        )

    cocotb.log.info("test_bias_sram_basic_addresses passed")


@cocotb.test()
async def test_bias_sram_full_sweep(dut):
    await init_dut(dut)
    for addr in range(151):
        got = await read_bias(dut, addr)
        exp = EXPECTED_HEX[addr] if addr < len(EXPECTED_HEX) else 0

        assert got == exp, (
            f"addr={addr}: expected {exp} (0x{(exp & 0xFFFFFFFF):08X}), "
            f"got {got} (0x{(got & 0xFFFFFFFF):08X})"
        )

    cocotb.log.info("test_bias_sram_full_sweep passed")


@cocotb.test()
async def test_bias_sram_block_boundaries(dut):
    await init_dut(dut)
    # 16-filter DS-CNN: first_conv(16) + 4 ds_blocks(16 dw + 16 pw each) + classifier(7)
    # bias offsets: 0-15, 16-31, 32-47, 48-63, 64-79, 80-95, 96-111, 112-127, 128-143, 144-150
    boundary_addrs = [
        0, 15,       # first_conv
        16, 31,      # ds_blocks.0.depthwise
        32, 47,      # ds_blocks.0.pointwise
        48, 63,      # ds_blocks.1.depthwise
        64, 79,      # ds_blocks.1.pointwise
        80, 95,      # ds_blocks.2.depthwise
        96, 111,     # ds_blocks.2.pointwise
        112, 127,    # ds_blocks.3.depthwise
        128, 143,    # ds_blocks.3.pointwise
        144, 150,    # classifier
    ]

    for addr in boundary_addrs:
        got = await read_bias(dut, addr)
        exp = EXPECTED_HEX[addr] if addr < len(EXPECTED_HEX) else 0

        assert got == exp, (
            f"addr={addr}: expected {exp} (0x{(exp & 0xFFFFFFFF):08X}), "
            f"got {got} (0x{(got & 0xFFFFFFFF):08X})"
        )

    cocotb.log.info("test_bias_sram_block_boundaries passed")
