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

- `pipeline_top.sv`: Top-level RTL 
- `test_pipeline_top.py`: cocotb testbench
- `compare_outputs.py`: 3-way feature comparison

## TODO: spect_buffer_ctrl (module between pipeline and dscnn)

-Accepts INT16 mel stream from output_buffer (cnn_data_ol/cnn_valid_ol) and writes INT8 values into the active spectrogram_sram bank
-Counts 50 frames of 40 mel bins each before signaling completion
-Pulses spect_done for one cycle after the 50th frame so FSM can begin inference
-Quantizes each value INT16→INT8 via arithmetic right shift by QUANT_SHIFT + saturating clamp to [-128, 127]
-On spect_done, flips spect_write_sel to swap which bank the preprocessor writes into next
