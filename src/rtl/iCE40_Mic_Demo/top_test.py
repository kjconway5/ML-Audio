import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
import math

CLK_PERIOD_NS = 10  # 100 MHz clock
FS_IN = 1_008_000   # PDM sample rate
TONE_FREQ = 1000    # 1 kHz audio tone


class SigmaDeltaPDM:
    def __init__(self):
        self.integrator = 0.0

    def step(self, x):
        # 1-bit quantizer
        y = 1.0 if self.integrator >= 0 else -1.0

        # integrate error
        self.integrator += x - y

        return 1 if y > 0 else 0


@cocotb.test()
async def test_cic_fir_chain(dut):

    # Start clock
    cocotb.start_soon(Clock(dut.clk, CLK_PERIOD_NS, units="ns").start())

    # Reset
    dut.rst.value = 1
    dut.pdm_sample.value = 0
    dut.pdm_valid.value = 0

    for _ in range(50):
        await RisingEdge(dut.clk)

    dut.rst.value = 0

    # Create PDM modulator
    pdm = SigmaDeltaPDM()

    phase = 0.0
    phase_inc = 2 * math.pi * TONE_FREQ / FS_IN

    output_samples = []

    for i in range(20000):  # run long enough for decimation
        await RisingEdge(dut.clk)

        # Generate PCM sine wave (-0.8 to 0.8)
        pcm = 0.8 * math.sin(phase)
        phase += phase_inc

        # Convert to PDM bit
        pdm_bit = pdm.step(pcm)

        # Convert to signed format expected by your design
        dut.pdm_sample.value = 1 if pdm_bit else -1
        dut.pdm_valid.value = 1

        # Capture output (16 kHz domain)
        if dut.audio_valid.value:
            val = int(dut.audio_out.value.signed_integer)
            output_samples.append(val)

    assert len(output_samples) > 50, "No output samples!"

    # Check dynamic range
    peak = max(output_samples)
    trough = min(output_samples)

    assert peak != trough, "Flat output (filter broken)"

    print(f"Samples captured: {len(output_samples)}")
    print(f"Peak: {peak}, Trough: {trough}")

    # Optional: crude frequency sanity check
    # Count zero crossings
    zero_crossings = 0
    for i in range(1, len(output_samples)):
        if output_samples[i-1] < 0 and output_samples[i] >= 0:
            zero_crossings += 1

    print(f"Zero crossings: {zero_crossings}")