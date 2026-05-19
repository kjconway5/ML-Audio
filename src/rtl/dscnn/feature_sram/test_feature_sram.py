# test_feature_sram.py
# Cocotb testbench for feature_sram.sv

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge
import os

CLK_PERIOD_NS = 100
DEPTH         = 8000
ADDR_W        = 14
NUM_BANKS     = 8

def test_pattern(addr):
    return (addr) % 256


async def init_dut(dut):
    """
    Interact with all 16 macro instances in each bank A and B so CEN sees
    a 1->0
    """
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())

    dut.a_we.value    = 0
    dut.a_waddr.value = 0
    dut.a_wdata.value = 0
    dut.a_raddr.value = 0
    dut.b_we.value    = 0
    dut.b_waddr.value = 0
    dut.b_wdata.value = 0
    dut.b_raddr.value = 0

    await RisingEdge(dut.clk)

    # Interact bank A
    for bank in range(NUM_BANKS):
        dut.a_raddr.value = bank * 1024
        await RisingEdge(dut.clk)
        await RisingEdge(dut.clk)

    # Interact bank B
    for bank in range(NUM_BANKS):
        dut.b_raddr.value = bank * 1024
        await RisingEdge(dut.clk)
        await RisingEdge(dut.clk)

    dut.a_raddr.value = 0
    dut.b_raddr.value = 0
    await RisingEdge(dut.clk)


async def write_bank_a(dut, start_addr=0, count=DEPTH):
    for addr in range(start_addr, start_addr + count):
        await FallingEdge(dut.clk)
        dut.a_we.value    = 1
        dut.a_waddr.value = addr
        dut.a_wdata.value = test_pattern(addr)
        await RisingEdge(dut.clk)

    await FallingEdge(dut.clk)
    dut.a_we.value    = 0
    dut.a_waddr.value = 0
    dut.a_wdata.value = 0
    await RisingEdge(dut.clk)


async def write_bank_b(dut, start_addr=0, count=DEPTH):
    for addr in range(start_addr, start_addr + count):
        await FallingEdge(dut.clk)
        dut.b_we.value    = 1
        dut.b_waddr.value = addr
        dut.b_wdata.value = test_pattern(addr)
        await RisingEdge(dut.clk)

    await FallingEdge(dut.clk)
    dut.b_we.value    = 0
    dut.b_waddr.value = 0
    dut.b_wdata.value = 0
    await RisingEdge(dut.clk)


async def read_bank_a(dut, addr):
    await FallingEdge(dut.clk)
    dut.a_raddr.value = addr
    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    return int(dut.a_rdata.value)


async def read_bank_b(dut, addr):
    await FallingEdge(dut.clk)
    dut.b_raddr.value = addr
    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    return int(dut.b_rdata.value)



@cocotb.test()
async def test_bank_a_write_read(dut):
    await init_dut(dut)
    await write_bank_a(dut)

    dut._log.info("Reading back all 16,000 entries from bank A...")
    errors = 0
    for addr in range(DEPTH):
        result   = await read_bank_a(dut, addr)
        expected = test_pattern(addr)
        if result != expected:
            dut._log.error(
                f"FAIL bank A addr={addr:5d} (0x{addr:04X}): "
                f"expected 0x{expected:02X} got 0x{result:02X}"
            )
            errors += 1
            if errors >= 10:
                raise AssertionError(f"Stopping after 10 errors. Last at addr={addr}")

    if errors == 0:
        dut._log.info(f"PASS: all {DEPTH} bank A addresses verified correctly")
    else:
        raise AssertionError(f"{errors} mismatches in bank A readback")


@cocotb.test()
async def test_bank_b_write_read(dut):
    await init_dut(dut)
    await write_bank_b(dut)

    dut._log.info("Reading back all 16,000 entries from bank B...")
    errors = 0
    for addr in range(DEPTH):
        result   = await read_bank_b(dut, addr)
        expected = test_pattern(addr)
        if result != expected:
            dut._log.error(
                f"FAIL bank B addr={addr:5d} (0x{addr:04X}): "
                f"expected 0x{expected:02X} got 0x{result:02X}"
            )
            errors += 1
            if errors >= 10:
                raise AssertionError(f"Stopping after 10 errors. Last at addr={addr}")

    if errors == 0:
        dut._log.info(f"PASS: all {DEPTH} bank B addresses verified correctly")
    else:
        raise AssertionError(f"{errors} mismatches in bank B readback")


@cocotb.test()
async def test_simultaneous_ab(dut):
    """
    Write to bank A while simultaneously reading from bank B.
    """
    await init_dut(dut)

    # Write B with a regular pattern
    await write_bank_b(dut)

    dut._log.info("Writing different pattern to bank A simultaneously with bank B reads...")

    errors = 0
    for addr in range(DEPTH):
        await FallingEdge(dut.clk)

        # Write to bank A
        dut.a_we.value    = 1
        dut.a_waddr.value = addr
        dut.a_wdata.value = (addr * 128) % 256

        # Simultaneously read from bank B
        dut.b_we.value    = 0
        dut.b_raddr.value = addr

        await RisingEdge(dut.clk)
        await FallingEdge(dut.clk)

        # Check bank B read is correct
        b_result   = int(dut.b_rdata.value)
        b_expected = test_pattern(addr)

        if b_result != b_expected:
            dut._log.error(
                f"FAIL simultaneous: bank B addr={addr:5d} "
                f"expected 0x{b_expected:02X} got 0x{b_result:02X} "
                f"while writing bank A"
            )
            errors += 1
            if errors >= 10:
                raise AssertionError(f"Stopping after 10 errors. Last at addr={addr}")

    dut.a_we.value = 0
    await RisingEdge(dut.clk)

    if errors == 0:
        dut._log.info(f"PASS: bank B reads unaffected by simultaneous bank A writes")
    else:
        raise AssertionError(f"{errors} bank B corruption errors during simultaneous bank A write")
