# test_weight_sram.py

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge
from cocotb.handle import Force, Release
import os

CLK_PERIOD_NS = 100  

WEIGHTS_HEX = os.path.join(os.path.dirname(__file__), "weights.hex")
NUM_WEIGHTS  = 6752


def load_weights_hex(path):
    """
    Load weights.hex into a list 
    """
    weights = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            weights.append(int(line, 16))
    assert len(weights) == NUM_WEIGHTS, \
        f"Expected {NUM_WEIGHTS} weights, got {len(weights)}"
    return weights


async def init_dut(dut):
    """"Initialize values. Initialize cen_fell to allow for valid writes"""
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())

    dut.we.value    = 0
    dut.waddr.value = 0
    dut.wdata.value = 0
    dut.raddr.value  = 0

    await RisingEdge(dut.clk)

    # Touch an addr in all banks to assert cen_fell -> 1
    for bank in range(7):
        dut.raddr.value = bank * 1024   
        await RisingEdge(dut.clk)      
        await RisingEdge(dut.clk)      

    dut.raddr.value = 0
    await RisingEdge(dut.clk)


async def write_weights(dut, weights):
    """
    Write all weights cycle-by-cycle through the write port.
    Mimics SERV loading weights on startup before inference begins.
    One write per clock cycle.
    """
    dut._log.info(f"Writing {len(weights)} weights into SRAM...")

    for addr, val in enumerate(weights):
        await FallingEdge(dut.clk)
        dut.we.value    = 1
        dut.waddr.value = addr
        dut.wdata.value = val          
        await RisingEdge(dut.clk)    

    await FallingEdge(dut.clk)
    dut.we.value    = 0
    dut.waddr.value = 0
    dut.wdata.value = 0
    await RisingEdge(dut.clk)

    dut._log.info("Write complete.")


async def read_addr(dut, addr):
    """
    Read one address. Present addr on falling edge, data valid 1 cycle later.
    Returns unsigned integer 0-255.
    """
    await FallingEdge(dut.clk)
    dut.raddr.value = addr
    await RisingEdge(dut.clk)   
    await FallingEdge(dut.clk)  
    return int(dut.rdata.value)


@cocotb.test()
async def test_full_read(dut):
    """
    After writing all weights, read back every single address and
    compare against weights.hex. Full test that writes and reads
    all weight values. 
    """
    await init_dut(dut)

    weights = load_weights_hex(WEIGHTS_HEX)
    await write_weights(dut, weights)

    dut._log.info("Starting full sequential readback of all 4296 addresses...")

    errors = 0
    for addr in range(NUM_WEIGHTS):
        result   = await read_addr(dut, addr)
        expected = weights[addr]

        if result != expected:
            dut._log.error(
                f"FAIL addr={addr:4d} (0x{addr:04X}): "
                f"expected 0x{expected:02X}, got 0x{result:02X}"
            )
            errors += 1
            if errors >= 10:
                raise AssertionError(f"Stopping after 10 errors. Last failure at addr={addr}")

    if errors == 0:
        dut._log.info(f"PASS: all {NUM_WEIGHTS} addresses verified correctly")
    else:
        raise AssertionError(f"{errors} total address mismatches in full readback")


@cocotb.test()
async def test_read_during_write(dut):
    """
    Verify that if we=1 is asserted during a read cycle (e.g. unexpected SERV access),
    the read output is corrupted on that cycle but recovers correctly
    on the next cycle once we=0 is restored.
 
    This test shows how our read/write SRAM can only do one or the other,
    and shows how the SRAM would react to a simultaneous read/write (1 cycle corruption).
 
    In normal chip operation this never occurs — SERV completes all weight
    writes before asserting start, so FSM and SERV should never access simultaneously.
    """
    await init_dut(dut)
 
    weights = load_weights_hex(WEIGHTS_HEX)
    await write_weights(dut, weights)
 
    # Sanity check that SRAM read works correctly at addr=0 before the test
    baseline = await read_addr(dut, 0)
    assert baseline == weights[0], \
        f"Baseline failed: expected 0x{weights[0]:02X} got 0x{baseline:02X}"
    dut._log.info(f"Baseline read addr=0: 0x{baseline:02X} — correct")
    
    # Find corrupt_addr in a different bank from addr=0
    corrupt_addr = None
    for candidate in range(1, NUM_WEIGHTS):
        if weights[candidate] != weights[0] and (candidate >> 10) != 0:
            corrupt_addr = candidate
            break
 
    assert corrupt_addr is not None, \
        "Could not find two addresses with different values in weights.hex"
    dut._log.info(
        f"Selected addr=0 (0x{weights[0]:02X}) and "
        f"corrupt_addr={corrupt_addr} (0x{weights[corrupt_addr]:02X}) "
        f"for corruption test — values differ so corruption is detectable"
    )
 
    # Assert we=1 writing to corrupt_addr while read port has addr=0
    await FallingEdge(dut.clk)
    dut.we.value    = 1
    dut.waddr.value = corrupt_addr
    dut.wdata.value = weights[corrupt_addr]
    dut.raddr.value  = 0
 
    await RisingEdge(dut.clk)       
    await FallingEdge(dut.clk)
    corrupted = int(dut.rdata.value)
 
    dut._log.info(
        f"During we=1: data=0x{corrupted:02X} "
        f"(expected corruption — active_addr={corrupt_addr} not 0)"
    )
    assert corrupted != weights[0] or corrupted == weights[corrupt_addr], \
        f"Expected corruption but got clean read — check address mux"
 
    dut.we.value    = 0
    dut.waddr.value = 0
    dut.wdata.value = 0
    await RisingEdge(dut.clk)       
    await FallingEdge(dut.clk)
    recovered = int(dut.rdata.value)
 
    assert recovered == weights[0], \
        f"FAIL recovery: expected 0x{weights[0]:02X} got 0x{recovered:02X}"
    dut._log.info(
        f"PASS recovery: addr=0 reads 0x{recovered:02X} correctly after we deasserted"
    )