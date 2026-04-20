# tb/fft/test_fft_data_ram.py
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge
import random

async def reset(dut):
    dut.ract.value  = 0
    dut.wact.value  = 0
    dut.addr.value  = 0
    dut.wdata.value = 0
    await RisingEdge(dut.clk)

async def write(dut, addr, data):
    await FallingEdge(dut.clk)
    dut.wact.value  = 1
    dut.ract.value  = 0
    dut.addr.value  = addr
    dut.wdata.value = data
    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    dut.wact.value  = 0

async def read(dut, addr):
    await FallingEdge(dut.clk)
    dut.ract.value  = 1
    dut.wact.value  = 0
    dut.addr.value  = addr
    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)   # data valid one cycle after rising edge
    dut.ract.value  = 0
    return int(dut.rdata.value)

@cocotb.test()
async def test_write_read_back(dut):
    """Write known values to every address, read back and verify."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)

    test_data = {addr: random.randint(0, 0xFFFFFFFF) for addr in range(128)}

    # Write phase
    for addr, data in test_data.items():
        await write(dut, addr, data)

    # Read phase
    for addr, expected in test_data.items():
        got = await read(dut, addr)
        assert got == expected, (
            f"addr={addr:#04x}: expected {expected:#010x}, got {got:#010x}"
        )

@cocotb.test()
async def test_byte_lanes(dut):
    """Verify all 4 byte lanes are independent (catch wiring mistakes)."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)

    patterns = [0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000]
    for i, pat in enumerate(patterns):
        await write(dut, i, pat)
        got = await read(dut, i)
        assert got == pat, f"byte lane {i}: expected {pat:#010x}, got {got:#010x}"

@cocotb.test()
async def test_no_read_write_collision(dut):
    """Simultaneous ract+wact should not corrupt data — wact wins."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)

    await write(dut, 0, 0xDEADBEEF)
    got = await read(dut, 0)
    assert got == 0xDEADBEEF