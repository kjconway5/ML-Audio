#!/usr/bin/env python3


import sys
import os
from pathlib import Path

# Add util directory to path for utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "util"))

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, ClockCycles, Timer
from cocotb.regression import TestFactory

from utilities import clock_start_sequence, reset_sequence


@cocotb.test()
async def basic_counter_test(dut):
    """Test basic counter functionality"""
    
    
    # Start clock (1ns period = 1GHz)
    await clock_start_sequence(dut.clk_i, period=1, unit='ns')
    
    # Initialize
    dut.en_i.value = 0

    # Apply reset
    await reset_sequence(dut.clk_i, dut.reset_i, cycles=2, active_level=True)
    
    # Check counter is at 0
    await RisingEdge(dut.clk_i)
    assert dut.count_o.value == 0, f"Counter should be 0 after reset, got {dut.count_o.value}"
    dut._log.info("Reset test passed")
    
    # Enable counter (count up)
    await FallingEdge(dut.clk_i)
    dut.en_i.value = 1

    # Count for 10 cycles and verify
    # Counter increments on rising edge, so we need to wait for the edge
    # then sample on the falling edge (or after some delay)
    for i in range(1, 11):
        await RisingEdge(dut.clk_i)
        await FallingEdge(dut.clk_i)
        expected = i
        actual = int(dut.count_o.value)
        dut._log.info(f"Cycle {i}: count_o = {actual}")
        assert actual == expected, f"Expected {expected}, got {actual}"
    
    
    # Disable counter
    await FallingEdge(dut.clk_i)
    dut.en_i.value = 0
    
    # Verify counter holds value
    held_value = int(dut.count_o.value)
    await ClockCycles(dut.clk_i, 5)
    await RisingEdge(dut.clk_i)
    assert int(dut.count_o.value) == held_value, "Counter should hold value when disabled"
    


@cocotb.test()
async def overflow_test(dut):
    """Test counter overflow behavior"""
    
    
    # Start clock
    await clock_start_sequence(dut.clk_i, period=1, unit='ns')
    
    # Reset
    dut.en_i.value = 0
    await reset_sequence(dut.clk_i, dut.reset_i, cycles=2, active_level=True)

    # Enable and count to overflow
    await FallingEdge(dut.clk_i)
    dut.en_i.value = 1

    # Get the max value based on width_p parameter
    width = int(dut.width_p.value)
    max_val = (1 << width) - 1
    
    dut._log.info(f"Counter width: {width}, max value: {max_val}")
    
    # Count up to max value
    for _ in range(max_val):
        await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)

    # Verify we're at max
    assert int(dut.count_o.value) == max_val, f"Expected max value {max_val}"
    
    # One more clock should overflow to 0
    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    assert int(dut.count_o.value) == 0, "Counter should overflow to 0"
    


@cocotb.test()
async def reset_during_count_test(dut):
    """Test reset assertion during counting"""
    
    
    # Start clock
    await clock_start_sequence(dut.clk_i, period=1, unit='ns')
    
    # Initial reset
    dut.en_i.value = 0
    await reset_sequence(dut.clk_i, dut.reset_i, cycles=2, active_level=True)

    # Enable and count
    await FallingEdge(dut.clk_i)
    dut.en_i.value = 1
    
    # Count for a bit
    await ClockCycles(dut.clk_i, 15)
    
    current_val = int(dut.count_o.value)
    dut._log.info(f"Counter at {current_val} before reset")
    assert current_val > 0, "Counter should have incremented"
    
    # Apply reset mid-count
    await FallingEdge(dut.clk_i)
    dut.reset_i.value = 1
    await RisingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    dut.reset_i.value = 0
    
    # Verify reset
    await RisingEdge(dut.clk_i)
    assert int(dut.count_o.value) == 0, "Counter should reset to 0"
    
    # Verify it continues counting after reset
    await ClockCycles(dut.clk_i, 5)
    assert int(dut.count_o.value) == 5, "Counter should resume counting"
    


