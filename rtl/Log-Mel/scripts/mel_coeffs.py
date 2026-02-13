import numpy as np
import torchaudio
import matplotlib.pyplot as plt

SR         = 16000
N_FFT      = 256
N_MELS     = 40
FFT_BINS   = N_FFT // 2  # 128 useful bins
COEFF_W    = 16 # bits per weight in ROM
FRAC_BITS  = 15 # Q0.15 format: 1.0 → 32767

# generate mel filterbank
# Shape returned: (n_freqs, n_mels) = (129, 40)
# n_freqs = N_FFT//2 + 1 = 129
mel_fb = torchaudio.functional.melscale_fbanks(
    n_freqs=N_FFT // 2 + 1,
    f_min=0.0,
    f_max=SR / 2.0,
    n_mels=N_MELS,
    sample_rate=SR,
    norm=None, 
    mel_scale='htk'
)
# Transpose and drop bin 128 → shape (40, 128)
mel_fb = mel_fb[:FFT_BINS, :].T.numpy()   # (40, 128)

# Quantize to Q0.15 fixed point
mel_fixed = np.round(mel_fb * (2**FRAC_BITS)).astype(np.int32)
mel_fixed = np.clip(mel_fixed, 0, 2**COEFF_W - 1)

# Find sparse ranges per mel filter
start_bins = []
end_bins   = []
all_coeffs = []

for m in range(N_MELS):
    nonzero = np.where(mel_fixed[m] > 0)[0]
    if len(nonzero) == 0:
        start_bins.append(0)
        end_bins.append(0)
        all_coeffs.append([0])
    else:
        start = int(nonzero[0])
        end   = int(nonzero[-1])
        start_bins.append(start)
        end_bins.append(end)
        all_coeffs.append(mel_fixed[m, start:end+1].tolist())

max_coeffs = max(len(c) for c in all_coeffs)

print(f"N_FFT={N_FFT}, N_MELS={N_MELS}, FFT_BINS={FFT_BINS}")
print(f"Max non-zero weights per filter : {max_coeffs}")
print(f"Sparse ROM size : "
      f"{N_MELS * max_coeffs * COEFF_W // 8} bytes")
print(f"Dense ROM size  : "
      f"{N_MELS * FFT_BINS * COEFF_W // 8} bytes")

# Pad all coefficient arrays to max_coeffs length
coeffs_padded = np.zeros((N_MELS, max_coeffs), dtype=np.int32)
for m in range(N_MELS):
    c = all_coeffs[m]
    coeffs_padded[m, :len(c)] = c

# Convert bin indices to Hz for x axis
bin_hz = np.linspace(0, SR/2, FFT_BINS)

plt.figure(figsize=(12, 4))
for m in range(N_MELS):
    plt.plot(bin_hz, mel_fb[m])

plt.xlabel("Frequency (Hz)")
plt.ylabel("Filter weight")
plt.title("Mel Filterbank (40 filters, N_FFT=256, 16kHz)")
plt.tight_layout()
plt.show()

# Also print what you probably expected to see
print("Filter peaks in Hz:")
for m in range(10):
    peak_bin = np.argmax(mel_fb[m])
    peak_hz  = bin_hz[peak_bin]
    print(f"  Mel {m:2d}: peak at bin {peak_bin:3d} = {peak_hz:.1f} Hz, "
          f"weight = {mel_fb[m, peak_bin]:.4f}")

# ── Write hex files ────────────────────────────────────────────
# mel_coeffs.hex — flattened row by row (mel0_coeff0, mel0_coeff1, ...)
with open("mel_coeffs.hex", "w") as f:
    for m in range(N_MELS):
        for k in range(max_coeffs):
            f.write(f"{coeffs_padded[m, k] & 0xFFFF:04x}\n")

# mel_starts.hex — start bin index per mel filter
with open("mel_starts.hex", "w") as f:
    for s in start_bins:
        f.write(f"{s:02x}\n")

# mel_ends.hex — end bin index per mel filter
with open("mel_ends.hex", "w") as f:
    for e in end_bins:
        f.write(f"{e:02x}\n")

print("\nWritten: mel_coeffs.hex, mel_starts.hex, mel_ends.hex")

# ── Sanity check ───────────────────────────────────────────────
print("\nFilter ranges (first 10):")
for m in range(min(10, N_MELS)):
    n_weights = end_bins[m] - start_bins[m] + 1
    peak = mel_fixed[m].max()
    print(f"  Mel {m:2d}: bins {start_bins[m]:3d}–{end_bins[m]:3d} "
          f"({n_weights:2d} weights, peak={peak})")

# ── Also generate log2 LUT while we're here ───────────────────
LOG_LUT_FRAC_BITS = 6    # 64-entry LUT
LOG_OUT_W         = 16
Q_FRAC            = 12   # Q4.12 output format

lut_size  = 2**LOG_LUT_FRAC_BITS
frac_vals = np.arange(lut_size) / lut_size   # 0, 1/64, 2/64, ...

# LUT stores log2(1 + frac) in Q4.12
lut_float = np.log2(1.0 + frac_vals)         # range: 0.0 to ~0.9999
lut_fixed = np.round(lut_float * (2**Q_FRAC)).astype(np.int32)

with open("log2_lut.hex", "w") as f:
    for val in lut_fixed:
        f.write(f"{val:04x}\n")

print(f"\nlog2 LUT: {lut_size} entries, Q4.12 format")
print(f"Written: log2_lut.hex")
print(f"LUT range: 0x{lut_fixed[0]:04x} to 0x{lut_fixed[-1]:04x}")