# ML-Audio Source Guide

## config.yaml

`ml/Pipeline/config.yaml` — read by `process_data.py` and `train.py`.

Set `dataset.data_dir` to the Speech Commands v0.02 root and `data.output_dir` to where processed `.npy` files should land.

**Output file names** are set under `output`:
```yaml
output:
  model_save_path: "/path/to/models/dscnn-32requant-v11/dscnn-32requant-v11.pt"
  log_file:        "/path/to/models/dscnn-32requant-v11/training_log_32requant_v11.txt"
```

Other notable parameters — these touch quantization/RTL behavior and shouldn't be changed without understanding the full pipeline:
- `preprocessing.*` — mel/FFT settings must stay in sync with RTL
- `model.*` — network size affects exported weight/bias layout
- `training.qat_start_epoch`, `freeze_bn_epoch` — control the float → QAT transition

Safe to tune freely: `training.n_epochs`, `batch_size`, `learning_rate`, `momentum`, `weight_decay`, `data.num_silence_samples`, `unknown_max_*`.

---

## Training a Model

```bash
cd src/ml/Pipeline/

# 1. Process raw audio into .npy feature arrays (one-time, ~5-10 min)
python3 process_data.py

# 2. Train (reads config.yaml — ~1 hr on GPU, QAT phases run on CPU)
python3 train.py

# 3. Export weights, biases, and per-layer shifts to RTL-ready files
python3 export.py --ckpt ../models/dscnn-32requant-v11/dscnn-32requant-v11.pt

# 4. Patch RTL files with the new weights/shifts (--dry-run to preview first)
python3 update_rtl.py --ckpt-dir ../models/dscnn-32requant-v11/ --dry-run
python3 update_rtl.py --ckpt-dir ../models/dscnn-32requant-v11/
```

`export.py` writes `weights.hex`, `bias.hex`, and `scales.txt` into the checkpoint directory.  
`update_rtl.py` patches `rtl/dscnn/kws_top/test_kws_top.py` (layer shifts), copies `bias.hex`, and retargets the bias SRAM parameters.

---

## Running the RTL Testbenches

### Generate spectrogram test vectors

```bash
cd src/rtl/dscnn/kws_top/

python3 generate_spect.py \
    --keyword yes \
    --n-samples 10 \
    --dataset-dir /path/to/speech_commands_v0.02 \
    --ckpt dscnn-32requant-v11/dscnn-32requant-v11.pt \
    --out-dir ./spectrograms
```

Other useful flags: `--wav-file <path>` to use a specific file, `--seed <int>` for reproducibility.

### Run simulations

```bash
# End-to-end inference testbench
cd src/rtl/dscnn/kws_top/
make test-cocotb

# Log-Mel feature extraction testbench
cd src/rtl/Log-Mel/rtl/log_top/
make test-cocotb

# Any individual submodule (mac_array, requant, bias_SRAM, etc.)
cd src/rtl/dscnn/<module>/
make test-cocotb
```

Use `make test-cocotb-verilator` for faster simulation, `make clean` to remove build artifacts.

After the kws_top test finishes, a summary is written to `src/rtl/dscnn/kws_top/kws_results.txt`:

```
keyword: yes   8/10 passed
----------------------------------------
[0] PASS  6de7e6c4_nohash_0.wav          RTL=yes
[1] FAIL  9c1e9ae5_nohash_1.wav          RTL=no   [model miss]
...
```

`[RTL bug]` means the hardware disagrees with the integer golden model — likely an RTL issue.  
`[model miss]` means both RTL and the golden model got it wrong — model accuracy issue.
