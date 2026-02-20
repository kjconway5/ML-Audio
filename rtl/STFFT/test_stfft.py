import cocotb
from cocotb.triggers import RisingEdge, ClockCycles, Timer
from cocotb.clock import Clock
from cocotb.binary import BinaryValue
import numpy as np
import logging
import os

async def feed_samples(dut, samples):
    for i, sample in enumerate(samples):
        dut.i_sample.value = int(sample)
        dut.i_ce.value = 1
        await RisingEdge(dut.i_clk)
        dut.i_ce.value = 0
        
        if i < len(samples) - 1:
            await ClockCycles(dut.i_clk, 22)

@cocotb.test()
async def test_stfft_basic(dut):
    """Test basic STFFT functionality with 50% overlap"""
    
    dut._log.setLevel(logging.INFO)
    
    clock = Clock(dut.i_clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset DUT
    dut.i_reset.value = 1
    await ClockCycles(dut.i_clk, 5)
    dut.i_reset.value = 0
    await ClockCycles(dut.i_clk, 5)
    

    FFT_SIZE = 256
    HOP_SIZE = 128  # 50% overlap
    SAMPLE_RATE = 10000  # Hz
    TEST_FREQ = 1000  # Hz
    
    # Generate 3 frames worth of samples
    num_samples = FFT_SIZE + HOP_SIZE * 2
    

    t = np.arange(num_samples) / SAMPLE_RATE
    test_signal = np.sin(2 * np.pi * TEST_FREQ * t)
    
    # Scale to 14-bit range (IW=14)
    max_val = 2**(14-1) - 1
    test_signal_scaled = (test_signal * max_val * 0.9).astype(int)
    
    dut._log.info(f"Generated {len(test_signal_scaled)} samples")
    dut._log.info(f"Signal range: min={min(test_signal_scaled)}, max={max(test_signal_scaled)}")
    
    # Initialize DUT
    dut.i_sample.value = 0
    dut.i_ce.value = 0
    
    # Feed samples to DUT
    dut._log.info("Starting to feed samples...")
    await feed_samples(dut, test_signal_scaled)
    
    # Wait for FFT processing
    await ClockCycles(dut.i_clk, FFT_SIZE * 3)
    
    # Monitor FFT outputs
    fft_count = 0
    max_ffts_to_capture = 3
    fft_results = []
    
    dut._log.info("Monitoring FFT outputs...")
    
    # Capture multiple FFT outputs
    for frame_num in range(max_ffts_to_capture):
        # Wait for sync signal with timeout
        sync_detected = False
        for _ in range(FFT_SIZE * 2):  # Timeout after 2 frames
            if dut.o_fft_sync.value:
                sync_detected = True
                break
            await RisingEdge(dut.i_clk)
        
        if sync_detected:
            dut._log.info(f"FFT sync detected for frame {fft_count + 1}")
            
            # Read FFT result
            fft_result = dut.o_fft_result.value
            
            # Store result
            if fft_result is not None:
                # Convert to integer
                if isinstance(fft_result, cocotb.binary.BinaryValue):
                    result_int = int(fft_result)
                else:
                    result_int = int(fft_result)
                
                fft_results.append(result_int)
                
                # Extract 18-bit real and imaginary parts (OW=18)
                mask = (1 << 18) - 1
                real_part = result_int & mask
                imag_part = (result_int >> 18) & mask
                
                # Convert from 2's complement if needed
                if real_part & (1 << 17):
                    real_part = real_part - (1 << 18)
                if imag_part & (1 << 17):
                    imag_part = imag_part - (1 << 18)
                
                # Calculate magnitude
                magnitude = np.sqrt(real_part**2 + imag_part**2)
                
                dut._log.info(f"Frame {fft_count + 1}: Real={real_part}, Imag={imag_part}, Mag={magnitude:.2f}")
            
            fft_count += 1
        else:
            dut._log.warning(f"Timeout waiting for FFT sync for frame {frame_num + 1}")
        
        # Wait a bit before looking for next sync
        await ClockCycles(dut.i_clk, 10)
    
    # Basic assertions
    assert fft_count > 0, "No FFT outputs received"
    dut._log.info(f"Successfully captured {fft_count} FFT frames")
    
    # With 50% overlap and 3 frames of samples, we should get at least 2 FFT outputs
    assert fft_count >= 2, f"Expected at least 2 FFT frames, got {fft_count}"
    
    # Log summary
    dut._log.info(f"Captured {len(fft_results)} FFT results")
    dut._log.info("Test completed successfully")

@cocotb.test()
async def test_stfft_sine_analysis(dut):
    """Test STFFT with sine wave and verify frequency content"""
    
    # Setup logging
    dut._log.setLevel(logging.INFO)
    
    # Start clock
    clock = Clock(dut.i_clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset DUT
    dut.i_reset.value = 1
    await ClockCycles(dut.i_clk, 5)
    dut.i_reset.value = 0
    await ClockCycles(dut.i_clk, 5)
    
    # Test parameters
    FFT_SIZE = 256
    SAMPLE_RATE = 10000  # Hz
    TEST_FREQ = 1000  # Hz
    
    # Generate one frame of samples
    t = np.arange(FFT_SIZE) / SAMPLE_RATE
    test_signal = np.sin(2 * np.pi * TEST_FREQ * t)
    
    # Scale to 14-bit range
    max_val = 2**(14-1) - 1
    test_signal_scaled = (test_signal * max_val * 0.9).astype(int)
    
    dut._log.info(f"Generated sine wave at {TEST_FREQ} Hz")
    
    # Initialize DUT
    dut.i_sample.value = 0
    dut.i_ce.value = 0
    
    # Feed samples
    await feed_samples(dut, test_signal_scaled)
    
    # Wait for FFT processing
    await ClockCycles(dut.i_clk, FFT_SIZE * 2)
    
    # Wait for sync
    sync_detected = False
    for _ in range(FFT_SIZE * 2):
        if dut.o_fft_sync.value:
            sync_detected = True
            break
        await RisingEdge(dut.i_clk)
    
    assert sync_detected, "No FFT sync detected"
    
    # Read result
    fft_result = dut.o_fft_result.value
    
    if fft_result is not None:
        if isinstance(fft_result, cocotb.binary.BinaryValue):
            result_int = int(fft_result)
        else:
            result_int = int(fft_result)
        
        # Extract real and imaginary
        mask = (1 << 18) - 1
        real_part = result_int & mask
        imag_part = (result_int >> 18) & mask
        
        # Convert from 2's complement
        if real_part & (1 << 17):
            real_part = real_part - (1 << 18)
        if imag_part & (1 << 17):
            imag_part = imag_part - (1 << 18)
        
        magnitude = np.sqrt(real_part**2 + imag_part**2)
        
        dut._log.info(f"FFT Output - Real: {real_part}, Imag: {imag_part}, Magnitude: {magnitude:.2f}")
        
        # For a sine wave input, we expect non-zero magnitude
        assert magnitude > 100, f"FFT magnitude too small: {magnitude}"
    
    dut._log.info("Sine wave analysis test completed")

# Note: No pytest function here - the test runner will handle this