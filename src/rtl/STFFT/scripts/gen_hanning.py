# scripts/gen_hanning.py — regenerate for IW=16, N=256
import numpy as np

N = 256
k = np.arange(N)
w = 0.5 * (1 - np.cos(2 * np.pi * k / N))
w_scaled = np.round(w * 32767).astype(np.int16)

with open("/workspace/src/rtl/STFFT/ZipCPU/hanning.hex", "w") as f:
    for v in w_scaled:
        f.write(f"{int(v) & 0xFFFF:04x}\n")  # cast to Python int first

print(f"k=0:   {w_scaled[0]:6d}  (expected 0)")
print(f"k=64:  {w_scaled[64]:6d}  (expected ~16384)")
print(f"k=128: {w_scaled[128]:6d}  (expected 32767)")
print(f"k=192: {w_scaled[192]:6d}  (expected ~16384)")
print(f"k=255: {w_scaled[255]:6d}  (expected ~40)")