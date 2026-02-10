import cocotb
from cocotb.triggers import RisingEdge, Timer
from cocotb.clock import Clock
import numpy as np

@cocotb.test()
async def test_stfft_basic(dut):
    """Test STFFT with a known sine wave"""

    # Clock
    cocotb.start_soon(Clock(dut.i_clk, 10, units="ns").start())

    # Reset
    dut.i_reset.value = 1
    dut.i_ce.value = 0
    dut.i_sample.value = 0
    await RisingEdge(dut.i_clk)
    await RisingEdge(dut.i_clk)
    dut.i_reset.value = 0

    FFT_SIZE = 256
    HOP_SIZE = 128

    # Generate test input: 1 kHz sine sampled at 16 kHz
    fs = 16_000
    f = 1_000
    t = np.arange(FFT_SIZE*2)/fs  # 2 frames
    x = (np.sin(2*np.pi*f*t) * (2**15-1)).astype(int)  # scale to 16-bit

    idx = 0
    n_samples = len(x)

    fft_outputs = []

    while idx < n_samples:
        dut.i_sample.value = int(x[idx])
        dut.i_ce.value = 1
        await RisingEdge(dut.i_clk)

        if dut.o_fft_sync.value:
            # Grab the FFT output when sync asserts
            fft_val = int(dut.o_fft_result.value)
            fft_outputs.append(fft_val)

        idx += 1

    # Optional: compare against reference FFT
    ref_fft = np.fft.fft(x[:FFT_SIZE])
    print("Reference FFT first bin:", ref_fft[0])
    print("DUT FFT outputs captured:", len(fft_outputs))
