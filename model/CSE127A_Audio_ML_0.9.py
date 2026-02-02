import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np

#globals

PDM_CLK = 1_008_000          # Hz
FS = 16_000                 # Hz
DECIM = 63                  # CIC decimation
CIC_STAGES = 5

# STFT
N_FFT = 512
WIN_MS = 25e-3
HOP_MS = 10e-3
WIN_LEN = int(FS * WIN_MS)  # 400
HOP_LEN = int(FS * HOP_MS)  # 160

# Mel
N_MELS = 40



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
        # Convert to bipolar
        x = pdm_bits.float() * 2.0 - 1.0

        # Integrators (PDM rate)
        for _ in range(self.stages):
            x = torch.cumsum(x, dim=0)

        # Decimate
        x = x[::self.decim]

        # Combs (audio rate)
        for _ in range(self.stages):
            x = torch.cat([x[:1], x[1:] - x[:-1]])

        return x



# CIC Compensation FIR 
def design_cic_comp_fir_hardware(
    num_taps=15,          # MUST be odd
    fs=FS,
    decim=DECIM,
    stages=CIC_STAGES,
    passband_hz=7000,     # speech band
    coeff_bits=16         # FIR coefficient width
):
    assert num_taps % 2 == 1, "FIR length must be odd"

    # Frequency grid (audio rate)
    f = np.linspace(0, fs / 2, 2048)

    # CIC magnitude
    eps = 1e-12
    num = np.sin(np.pi * f * decim / fs)
    den = np.sin(np.pi * f / fs) + eps
    cic_mag = np.abs(num / den) ** stages
    cic_mag /= cic_mag.max()

    # Desired response (inverse CIC in passband)
    desired = np.ones_like(f)
    pb = f <= passband_hz
    desired[pb] = 1.0 / np.maximum(cic_mag[pb], 1e-3)
    desired[~pb] = 0.0

    # Least-squares FIR (linear phase)
    M = (num_taps - 1) // 2
    A = np.zeros((len(f), M + 1))

    for k in range(M + 1):
        A[:, k] = np.cos(2 * np.pi * f * k / fs)

    h_ls, *_ = np.linalg.lstsq(A, desired, rcond=None)

    # Build symmetric impulse response
    h = np.zeros(num_taps)
    h[M] = h_ls[0]
    for k in range(1, M + 1):
        h[M + k] = h[M - k] = h_ls[k] / 2

    # Normalize DC gain
    h /= np.sum(h)

    # Quantize to fixed-point
    scale = 2 ** (coeff_bits - 1)
    h_q = np.round(h * scale).astype(int)

    # Renormalize exactly (hardware-style)
    h_q[M] += scale - np.sum(h_q)

    return h_q, scale



class FIRFilterHardware(nn.Module):
    def __init__(self, taps_int, scale):
        super().__init__()
        self.register_buffer(
            "taps",
            torch.tensor(taps_int, dtype=torch.int64)
        )
        self.scale = scale
        self.L = len(taps_int)

    def forward(self, x):
        x = x.to(torch.int64)

        y = torch.zeros_like(x)
        for n in range(self.L, len(x)):
            acc = 0
            for k in range(self.L):
                acc += x[n - k] * self.taps[k]
            y[n] = acc // self.scale  # fixed-point rounding

        return y.to(torch.float32)



def fifo_buffer(x):
    return x



# STFT + Power Spectrogram

window = torch.hann_window(WIN_LEN)

def stft_block(x):
    return torch.stft(
        x,
        n_fft=N_FFT,
        hop_length=HOP_LEN,
        win_length=WIN_LEN,
        window=window,
        center=False,
        return_complex=True
    )

def power_spectrogram(X):
    return torch.abs(X) ** 2


# Log-Mel Spectrogram

mel_filterbank = torchaudio.transforms.MelScale(
    n_mels=N_MELS,
    sample_rate=FS,
    n_stft=N_FFT // 2 + 1
)

def log_mel_spectrogram(power_spec):
    mel = mel_filterbank(power_spec)
    return torch.log(mel + 1e-6)



# Frontend

class AudioFrontend(nn.Module):
    def __init__(self):
        super().__init__()
        self.cic = CICDecimator()
        taps = design_cic_comp_fir_hardware()
        self.fir = FIRFilterHardware(taps)

    def forward(self, pdm_bits):
        x = self.cic(pdm_bits)
        x = self.fir(x)
        x = fifo_buffer(x)

        X = stft_block(x)
        power = power_spectrogram(X)
        logmel = log_mel_spectrogram(power)

        # Shape for CNN: [B, C, MELS, TIME]
        logmel = logmel.unsqueeze(0).unsqueeze(0)
        return logmel


# Depthwise-Separable CNN 
class Depthwise_Piecewise_Conv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
    ):
        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            groups=in_channels,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.depthwise(x)))
        x = F.relu(self.bn2(self.pointwise(x)))
        return x


class DSCNN(nn.Module):
    def __init__(
        self,
        n_classes=1,
        first_conv_filters=64,
        first_conv_kernel=(10, 4),
        first_conv_stride=(2, 2),
        n_ds_blocks=4,
        ds_filters=64,
    ):
        super().__init__()

        pad = ((first_conv_kernel[0] - 1) // 2,
               (first_conv_kernel[1] - 1) // 2)

        self.first_conv = nn.Conv2d(
            1,
            first_conv_filters,
            first_conv_kernel,
            first_conv_stride,
            pad,
            bias=False,
        )
        self.first_bn = nn.BatchNorm2d(first_conv_filters)

        self.ds_blocks = nn.ModuleList()
        for i in range(n_ds_blocks):
            in_ch = first_conv_filters if i == 0 else ds_filters
            self.ds_blocks.append(
                Depthwise_Piecewise_Conv(in_ch, ds_filters)
            )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(ds_filters, n_classes)

    def forward(self, x):
        x = F.relu(self.first_bn(self.first_conv(x)))
        for blk in self.ds_blocks:
            x = blk(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)



# Test

if __name__ == "__main__":
    seconds = 1.0
    pdm_samples = int(PDM_CLK * seconds)
    pdm_stream = torch.randint(0, 2, (pdm_samples,))

    frontend = AudioFrontend()
    features = frontend(pdm_stream)

    print("Log-Mel shape:", features.shape)

    model = DSCNN(n_classes=1)
    out = model(features)

    print("Model output shape:", out.shape)
