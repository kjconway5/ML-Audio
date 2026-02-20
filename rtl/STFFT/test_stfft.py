import cocotb
from cocotb.triggers import RisingEdge, ClockCycles
from cocotb.clock import Clock
import numpy as np
import pytest
from cocotb_test.simulator import run


async def reset_dut(dut):
    dut.i_reset.value = 1
    await ClockCycles(dut.i_clk, 5)
    dut.i_reset.value = 0
    await ClockCycles(dut.i_clk, 5)

@cocotb.test()
async def test_stfft_basic(dut):
    """Test basic STFFT functionality"""
    clock = Clock(dut.i_clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    

    await reset_dut(dut)

    dut.i_ce.value = 0
    dut.i_sample.value = 0
    
    # Test parameters
    fft_size = 256 
    test_frequency = 1000  # Hz
    sample_rate = 10000  # Hz
    
    # Generate test signal (sine wave)
    t = np.arange(fft_size * 2) / sample_rate
    test_signal = np.sin(2 * np.pi * test_frequency * t)
    

    test_signal = (test_signal * 32767).astype(int)
    
    # Send samples with alternating CE pattern
    for i, sample in enumerate(test_signal):
        await RisingEdge(dut.i_clk)
        dut.i_sample.value = int(sample)
        
        
        if i % 2 == 0:
            dut.i_ce.value = 1
        else:
            dut.i_ce.value = 0
            
        await RisingEdge(dut.i_clk)
        dut.i_ce.value = 0 
    
    # Wait for processing
    await ClockCycles(dut.i_clk, fft_size * 3)
    
    # Check for output
    fft_result = dut.o_fft_result.value
    fft_sync = dut.o_fft_sync.value
    
    # Add assertions
    assert fft_sync is not None, "No FFT output received"
    
    cocotb.log.info("Test completed successfully")


@pytest.mark.parametrize("parameters", [{}])
def test_stfft_basic(parameters):
    """Run cocotb test with pytest"""
    run(
        verilog_sources=[
            "/workspace/rtl/STFFT/stfft.sv",
            "/workspace/rtl/STFFT/FFT/windowfn.v",
            "/workspace/rtl/STFFT/FFT/fftmain.v",
            "/workspace/rtl/STFFT/FFT/fftstage.v",
            "/workspace/rtl/STFFT/FFT/qtrstage.v",
            "/workspace/rtl/STFFT/FFT/laststage.v",
            "/workspace/rtl/STFFT/FFT/bitreverse.v",
            "/workspace/rtl/STFFT/FFT/hwbfly.v",
            "/workspace/rtl/STFFT/FFT/butterfly.v",
            "/workspace/rtl/STFFT/FFT/longbimpy.v",
            "/workspace/rtl/STFFT/FFT/bimpy.v",
            "/workspace/rtl/STFFT/FFT/convround.v"
        ],
        toplevel="stfft",
        module="test_stfft",
        parameters=parameters,
        sim_build="sim_build",
        waves=True 
    )