import torch
import torch.nn as nn
import torchaudio
import numpy as np

# Globals


# Clocks / Rates
PDM_CLK = 1_008_000          # Hz
FS = 16_000                 # Hz
DECIM = 63                  # 1.008 MHz / 16 kHz

# CIC
CIC_STAGES = 5

# STFT
N_FFT = 512
WIN_MS = 25e-3
HOP_MS = 10e-3
WIN_LEN = int(FS * WIN_MS)  # 400 samples
HOP_LEN = int(FS * HOP_MS)  # 160 samples

# Mel
N_MELS = 40

""""
def sigma_delta_pdm(x):
    integrator = 0.0
    pdm = np.zeros(len(x), dtype=np.int8)

    for i, s in enumerate(x):
        integrator += s
        if integrator >= 0:
            pdm[i] = 1
            integrator -= 1
        else:
            pdm[i] = 0
            integrator += 1

    return pdm
"""

# CIC Decimator

class CICDecimator(nn.Module):
    def __init__(self, decim=DECIM, stages=CIC_STAGES):
        super().__init__()
        self.decim = decim
        self.stages = stages

    def forward(self, pdm_bits):
        """
        pdm_bits: Tensor [N] with values {0,1}
        """
        # Convert to bipolar (+1/-1)
        x = pdm_bits.float()
        x = 2.0 * x - 1.0

        # Integrators (run at PDM clock)
        for _ in range(self.stages):
            x = torch.cumsum(x, dim=0)

        # Decimation
        x = x[::self.decim]

        # Combs (run at 16 kHz)
        for _ in range(self.stages):
            x = torch.cat([x[:1], x[1:] - x[:-1]])

        return x


# CIC Compensation FIR Filter
def design_cic_compensation_fir(
    num_taps=15,
    fs=FS,
    decim=DECIM,
    stages=CIC_STAGES
):
    nyq = fs / 2
    f = np.linspace(0, nyq, 2048)

    # CIC magnitude response
    sinc = np.sinc(f / (fs / decim))
    cic_mag = np.abs(sinc) ** stages
    cic_mag[cic_mag < 1e-6] = 1e-6

    # Invert passband
    desired = 1.0 / cic_mag
    desired[f > 0.45 * nyq] = 0.0

    # Windowed FIR approximation
    taps = np.hamming(num_taps) * desired[:num_taps]
    taps /= np.sum(taps)

    return torch.tensor(taps, dtype=torch.float32)


class FIRFilter(nn.Module):
    def __init__(self, taps):
        super().__init__()
        self.register_buffer("taps", taps.view(1, 1, -1))

    def forward(self, x):
        x = x.view(1, 1, -1)
        y = torch.nn.functional.conv1d(
            x,
            self.taps,
            padding=self.taps.size(-1) // 2
        )
        return y.squeeze()



# FIFO -> doesnt do anything just here for model
def fifo_buffer(x):
    """
    Hardware FIFO abstracted as pass-through
    """
    return x



# STFT + Magnitude
window = torch.hann_window(WIN_LEN)

def stft_block(x):
    X = torch.stft(
        x,
        n_fft=N_FFT,
        hop_length=HOP_LEN,
        win_length=WIN_LEN,
        window=window,
        center=False,
        return_complex=True
    )
    return X


def magnitude_spectrogram(X):
    return X.real**2 + X.imag**2



# Log-Mel Spectrogram

mel_filterbank = torchaudio.transforms.MelScale(
    n_mels=N_MELS,
    sample_rate=FS,
    n_stft=N_FFT // 2 + 1
)

def log_mel_spectrogram(mag):
    mel = mel_filterbank(mag)
    return torch.log(mel + 1e-6)



# Model
class AudioFrontend(nn.Module):
    def __init__(self):
        super().__init__()
        self.cic = CICDecimator()
        taps = design_cic_compensation_fir()
        self.fir = FIRFilter(taps)

    def forward(self, pdm_bits):
        x = self.cic(pdm_bits)
        x = self.fir(x)
        x = fifo_buffer(x)
        X = stft_block(x)
        mag = magnitude_spectrogram(X)
        logmel = log_mel_spectrogram(mag)
        return logmel



# test

if __name__ == "__main__":
    #1 second of fake PDM data
    seconds = 1.0
    num_pdm_samples = int(PDM_CLK * seconds)
    pdm_stream = torch.randint(0, 2, (num_pdm_samples,))

    frontend = AudioFrontend()
    features = frontend(pdm_stream)

    print("Log-Mel shape:", features.shape)
    # Expected: [n_mels, time_frames]
