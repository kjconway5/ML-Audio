# RTL Pipeline Verification

Verify the `pipeline_top` RTL (STFFT + Log-Mel) against software reference models.

## Usage

```bash
cd src/rtl/top
make test-cocotb            # simulate pipeline, saves rtl_features.npy
python3 compare_outputs.py  # compare RTL vs golden model vs torchaudio
```

## `make test-cocotb`

Runs [test_pipeline_top.py](test_pipeline_top.py) via cocotb + Icarus Verilog. Feeds a 200–7000 Hz chirp into `pipeline_top`, collects 54 output frames, and saves them to `rtl_features.npy`.

## `compare_outputs.py`

Regenerates the same chirp and runs it through:
- **Golden model** (`golden_model.py`) — bit-accurate fixed-point replica of RTL
- **torchaudio** (`features.py`) — floating-point reference

Prints delta statistics and saves a 4-panel spectrogram comparison to `comparison.png` (RTL, golden, torchaudio, delta heatmap).

## Files

- `pipeline_top.sv`: Top-level RTL (STFFT → logmel_top)
- `test_pipeline_top.py`: cocotb testbench
- `compare_outputs.py`: 3-way feature comparison
