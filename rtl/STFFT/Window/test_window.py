import cocotb
from cocotb.triggers import RisingEdge, ClockCycles
from cocotb.clock import Clock
import numpy as np
import torch

# Parameters
CLK_PERIOD_NS   = 10       # 100 MHz clock
WINDOW_LENGTH   = 256      # FFT_SIZE 
IW              = 14       # Input sample bit width 
OW              = 14       # Output sample bit width 
TW              = 14       # Tap/coefficient bit width 
SIGNED_MAX      = 2**(IW-1) - 1   # 8191 — used to normalize RTL output to float
SIGNED_MAX_TW   = 2**(TW-1) - 1   # 8191 — tap full-scale
CE_SPACING      = 23
FIRST_BLOCK     = WINDOW_LENGTH
TOLERANCE       = 1.0 / (2**(IW-2))  # ~4 LSBs of tolerance for fixed-point error

#
NORM_FACTOR = SIGNED_MAX * SIGNED_MAX_TW / (2 ** TW)

rtl_output_queue  = []
ref_output_queue  = []

#Trying to replicate windowing (Reference Model)
class My_window:
    def __init__(self, window_length):
        self.window = torch.hann_window(window_length, periodic=False)
        self.window_length = window_length

    def apply(self, frame):
        return self.window * frame

async def write_hann_taps(dut, window_length):
    hann = torch.hann_window(window_length, periodic=False).numpy()
    for val in hann:
        tap_int = int(round(val * SIGNED_MAX_TW))
        dut.i_tap_wr.value = 1
        dut.i_tap.value    = tap_int
        await RisingEdge(dut.i_clk)
    dut.i_tap_wr.value = 0
    await RisingEdge(dut.i_clk)

#Generates i_samples and pulses i_ce
async def driver(dut, num_frames = 10):
    ref_model = My_window(WINDOW_LENGTH)

    hop          = WINDOW_LENGTH // 2
    total_samples = FIRST_BLOCK + (num_frames * hop)

    # Generate a test signal: a sine wave at 1 kHz, sampled at 16 kHz
    # Scaled to signed 14-bit integer range
    t = np.arange(total_samples) / 16000.0
    test_signal = (np.sin(2 * np.pi * 1000 * t) * SIGNED_MAX).astype(np.int16)

    dut._log.info(f"Driver: sending {total_samples} samples "
                  f"({FIRST_BLOCK} first_block + {num_frames} frames)")

    # Pre-compute reference frames with 50% overlap
    # Frame k uses samples[k*hop : k*hop + WINDOW_LENGTH]
    all_float = (test_signal / SIGNED_MAX).astype(np.float32)
    for k in range(num_frames):
        start = k * hop
        frame_tensor = torch.tensor(
            all_float[start : start + WINDOW_LENGTH].copy(),
            dtype=torch.float32
        )
        ref_output_queue.append(ref_model.apply(frame_tensor))

    # Write Hann taps before driving any samples
    await write_hann_taps(dut, WINDOW_LENGTH)

    # Drive samples with alternating i_ce / i_alt_ce
    # Matches stfft.sv: alt_ce fires ~22 cycles after i_ce
    for sample in test_signal:
        #  Assert i_ce for 1 cycle
        dut.i_ce.value     = 1
        dut.i_sample.value = int(sample)
        await RisingEdge(dut.i_clk)

        # Deassert i_ce immediately
        dut.i_ce.value = 0

        # Wait, then pulse i_alt_ce (overlap read, no new data)
        # CE_SPACING - 2: one cycle consumed above, one for alt_ce below
        await ClockCycles(dut.i_clk, CE_SPACING - 2)

        dut.i_alt_ce.value = 1
        await RisingEdge(dut.i_clk)
        dut.i_alt_ce.value = 0

    dut._log.info("Driver: done sending samples")


#watch o_ce and capture o_sample whenever it is high.
async def monitor(dut, num_frames = 10):
    frames_captured = 0
    sample_buf      = []    # accumulates one frame worth of RTL output samples

    dut._log.info("Monitor: waiting for valid output (o_ce)")

    while frames_captured < num_frames:
        await RisingEdge(dut.i_clk)

        if dut.o_ce.value == 1:
            # Capture the signed OW-bit output and normalize to float
            # NORM_FACTOR accounts for the RTL's fixed-point gain so the
            # result is directly comparable to the Python model's output.
            raw = dut.o_sample.value.signed_integer
            normalized = raw / NORM_FACTOR
            sample_buf.append(normalized)

            # Once we have a full frame, deposit into the queue
            if len(sample_buf) == WINDOW_LENGTH:
                rtl_output_queue.append(
                    torch.tensor(sample_buf, dtype=torch.float32)
                )
                sample_buf      = []
                frames_captured += 1
                dut._log.info(f"Monitor: captured frame {frames_captured}/{num_frames}")

    dut._log.info("Monitor: done capturing frames")

#Checker coroutine
async def checker(dut, num_frames = 10):
    frames_checked = 0

    while frames_checked < num_frames:
        # Wait until both queues have a frame ready
        while len(rtl_output_queue) == 0 or len(ref_output_queue) == 0:
            await RisingEdge(dut.i_clk)

        rtl_frame = rtl_output_queue.pop(0)
        ref_frame = ref_output_queue.pop(0)

        # Compute error
        diff     = torch.abs(rtl_frame - ref_frame)
        max_err  = diff.max().item()
        rms_err  = torch.sqrt((diff**2).mean()).item()

        dut._log.info(
            f"Checker frame {frames_checked+1}/{num_frames}: "
            f"max_err={max_err:.6f}  rms_err={rms_err:.6f}  "
            f"tolerance={TOLERANCE:.6f}"
        )

        # Assert each sample is within tolerance
        for i, (rtl_val, ref_val, err) in enumerate(
                zip(rtl_frame.tolist(), ref_frame.tolist(), diff.tolist())):
            assert err <= TOLERANCE, (
                f"Frame {frames_checked+1}, sample {i}: "
                f"RTL={rtl_val:.6f}  REF={ref_val:.6f}  "
                f"err={err:.6f} exceeds tolerance={TOLERANCE:.6f}"
            )

        frames_checked += 1

    dut._log.info(f"Checker: all {num_frames} frames passed")

@cocotb.test()
async def test_windowfn(dut):

    NUM_FRAMES = 5   # how many frames to verify

    # Clear global queues (safe for re-runs)
    rtl_output_queue.clear()
    ref_output_queue.clear()

    #Start Clk
    cocotb.start_soon(Clock(dut.i_clk, CLK_PERIOD_NS, unit="ns").start())

    #Reset
    dut.i_reset.value  = 1
    dut.i_ce.value     = 0
    dut.i_sample.value = 0
    dut.i_tap_wr.value = 0
    dut.i_tap.value    = 0
    dut.i_alt_ce.value = 0
    await ClockCycles(dut.i_clk, 5)   # hold reset for 5 cycles
    dut.i_reset.value  = 0
    await RisingEdge(dut.i_clk)       # one clean cycle after reset

    # Launch driver and monitor concurrently
    cocotb.start_soon(driver(dut, num_frames=NUM_FRAMES))
    cocotb.start_soon(monitor(dut, num_frames=NUM_FRAMES))

    #Checker is awaited
    await checker(dut, num_frames=NUM_FRAMES)
