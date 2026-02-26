
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles
import numpy as np
import torchaudio.transforms as T

#Parameters 
N_MELS     = 40
N_BINS     = 129    
POWER_W    = 31
WEIGHT_W   = 16
ACCUM_W    = 54

SAMPLE_RATE = 16000
N_FFT       = 256
F_MIN       = 0.0
F_MAX       = SAMPLE_RATE / 2.0

POWER_MAX  = (1 << POWER_W) - 1
WEIGHT_MAX = (1 << WEIGHT_W) - 1
ACCUM_MASK = (1 << ACCUM_W) - 1

CLK_PERIOD_NS = 10  # 100 MHz

#Figure out how much leeway
TOLERANCE = 


#Reference Model
class My_MelFilterbank:


    def __init__(self):
        mel_t = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            n_mels=N_MELS,
            f_min=F_MIN,
            f_max=F_MAX,
            power=2.0
        )
        # fb: [N_BINS, N_MELS] float32
        self.fb = mel_t.mel_scale.fb.numpy()
        assert self.fb.shape == (N_BINS, N_MELS), \
            f"Unexpected filterbank {self.fb.shape}"

        # Quantise to Q0.15 
        fb_fixed = np.round(self.fb * (2 ** 15)).astype(np.int64)
        self.fb_fixed = np.clip(fb_fixed, 0, WEIGHT_MAX)   

        cocotb.log.info(
            f"MelFilterbank: torchaudio fb loaded, "
            f"max coeff={self.fb_fixed.max():#06x} ({self.fb_fixed.max()})"
        )

    def compute(self, power_bins: np.ndarray) -> np.ndarray:

        assert len(power_bins) == N_BINS
        p = (power_bins.astype(np.int64) & POWER_MAX)  

        # Matrix multiply
        accum = p @ self.fb_fixed                       
        return (accum & ACCUM_MASK).astype(np.uint64)


#
async def reset_dut(dut, cycles: int = 5):
    dut.reset_i.value  = 1
    dut.valid_il.value = 0
    dut.power_il.value = 0
    await ClockCycles(dut.clk_i, cycles)
    dut.reset_i.value = 0
    await RisingEdge(dut.clk_i)


async def drive_frame(dut, power_bins: np.ndarray):
    assert len(power_bins) == N_BINS
    for p in power_bins:
        dut.power_il.value = int(p) & POWER_MAX
        dut.valid_il.value = 1
        await RisingEdge(dut.clk_i)
    dut.valid_il.value = 0

# Wait for valid_ol to pulse high. 
async def wait_for_valid_ol(dut, timeout: int = N_BINS + 20) -> int:

    for i in range(timeout):
        await RisingEdge(dut.clk_i)
        if dut.valid_ol.value == 1:
            return i + 1
    raise AssertionError(f"valid_ol did not assert within {timeout} cycles")

# Capture all 40 mel_ol into array
def snapshot_outputs(dut) -> np.ndarray:
   
    return np.array([int(dut.mel_ol[m].value) for m in range(N_MELS)], dtype=np.uint64)

# Log mismatches and return the error count
def check_outputs(got: np.ndarray, expected: np.ndarray, label: str) -> int:

    errors = 0
    for m in range(N_MELS):
        delta = abs(int(got[m]) - int(expected[m]))
        if delta > TOLERANCE:
            cocotb.log.error(
                f"  {label} mel[{m:2d}]: "
                f"got={got[m]:#016x} ({got[m]}), "
                f"exp={expected[m]:#016x} ({expected[m]}), "
                f"delta={delta}  [tolerance={TOLERANCE}]"
            )
            errors += 1
        else:
            cocotb.log.debug(
                f"  {label} mel[{m:2d}]: {got[m]:#016x}  delta={delta} ✓"
            )
    return errors


