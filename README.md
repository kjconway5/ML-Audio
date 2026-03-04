# gf180mcu Project Template

Project template for wafer.space MPW runs using the gf180mcu PDK.

## Simulation & Verification

### Dependencies
```bash
pip install cocotb==1.9.2 cocotb-test==0.2.6 pytest==9.0.2 numpy torchaudio gitpython myhdl
```

System dependencies:
- Python 3.10+
- Icarus Verilog 11.0 - `sudo apt-get install iverilog` (Ubuntu/Debian)
- Verilator (optional, for lint and verilator-based simulation) - `sudo apt-get install verilator`
- GTKWave (optional, for waveform viewing) - `sudo apt-get install gtkwave`

### Repository Structure
```
src/
├── ml/                              # machine learning
│   ├── Pipeline/                    # training scripts and Python reference model
│   └── models/                      # trained model checkpoints
│
├── nngen/                           # NNGen CNN hardware generation
│   ├── 24fil-7class/                # 24 filter 7 class configuration
│   ├── ds-cnn-7-24/                 # DS-CNN NNGen project
│   ├── test/                        # NNGen test outputs
│   └── untrained/                   # untrained model baseline
│
└── rtl/                             # synthesizable RTL
    ├── CIC/                         # CIC decimation filter IP
    ├── FIFO/                        # 1R1W FIFO with handshake
    ├── FIR/                         # FIR anti-aliasing filter
    ├── STFFT/                       # STFT core
    │   └── ip/
    │       ├── FFT/                 # ZipCPU FFT IP
    │       │   ├── bench/cpp/       # C++ testbench
    │       │   └── bench/formal/    # formal verification
    │       └── Window/              # windowing logic
    ├── dscnn/                       # DS-CNN inference engine RTL
    └── Log-Mel/                     # log-mel filterbank
        ├── ip/                      # PULP arithmetic IP cores
        ├── data/                    # generated hex files (gitignored)
        ├── scripts/                 # mel coefficient generation
        └── rtl/
            ├── power_calc/          # Power Calculation - Re^2 + Im^2
            ├── mac_unit/            # Multiply-Accumulate
            ├── mel_filterbank/      # Filterbank
            ├── frame_control/       # Data pipeline FSM
            ├── log_lut/             # Log2 Compression
            ├── output_buffer/       # Mel Bin Output Buffer
            └── log_top/             # top-level integration test
```

### Makefile Targets

All modules written by our team share the same Makefile structure with these
targets:

| Target | Description |
|--------|-------------|
| `make lint` | Verilator lint check (design files only) |
| `make test` | Basic iverilog simulation |
| `make test-cocotb` | cocotb testbench with Icarus (default) |
| `make test-cocotb-icarus` | cocotb testbench with Icarus (explicit) |
| `make test-cocotb-verilator` | cocotb testbench with Verilator |
| `make wave` | Open waveform in GTKWave |
| `make clean` | Remove all build artifacts |

---

### Running Testbenches

---

#### FIR Filter
```bash
cd src/rtl/FIR
make test-cocotb
```

| Test | Description |
|------|-------------|
| TBD  | TBD |

---

#### CIC Decimation Filter

The CIC filter uses MyHDL cosimulation rather than cocotb. MyHDL drives the
SystemVerilog testbench `tb_cic.sv` via VPI (`$from_myhdl`/`$to_myhdl`).
```bash
cd src/rtl/CIC
python3 test_cic.py
```

A waveform dump is written to `test_cic_decimator.lxt`:
```bash
gtkwave test_cic_decimator.lxt
```

| Test | Description |
|------|-------------|
| test 1: impulse response | Verifies CIC impulse response at rate=2 |
| test 2: ramp | Ramp input at rate=2 |
| test 3: source pause | Ramp with input stream pausing intermittently |
| test 4: sink pause | Ramp with output sink pausing intermittently |
| test 5: sinewave | 1kHz sine wave decimation at rate=2 |
| test 6: rate of 4 | Sine wave decimation at rate=4 |
| test 7: DC | DC signal steady-state correctness |

---

#### STFFT

**Top-level cocotb test:**
```bash
cd src/rtl/STFFT
make test-cocotb
```

| Test | Description |
|------|-------------|
| `test_stfft_basic` | Drives a 1kHz sine wave at 16kHz sample rate, checks for valid FFT output sync |

**Window function:**
```bash
cd src/rtl/STFFT/ip/Window
make test-cocotb
```

| Test | Description |
|------|-------------|
| `test_windowfn` | Drives 5 frames of a 1kHz sine wave at 16kHz sample rate through the Hann window, captures RTL output and compares against a PyTorch reference model. Verifies each sample across all frames is within 4 LSBs of fixed-point tolerance |

**ZipCPU FFT IP: C++ testbenches:**
```bash
cd src/rtl/STFFT/ip/FFT/bench/cpp
# see README.md in this directory for build and run instructions
```

**ZipCPU FFT IP: Formal verification:**
```bash
cd src/rtl/STFFT/ip/FFT/bench/formal
# see README.md in this directory for tool requirements and run instructions
```

---

#### Log-Mel Filterbank

**Step 1: Generate hex files** (required before any log-mel simulation):
```bash
cd src/rtl/Log-Mel
python3 scripts/mel_coeffs.py
```

This writes to `src/rtl/Log-Mel/data/`:
- `mel_coeffs.hex` - Q0.15 sparse mel filter weights
- `mel_starts.hex` - start bin index per mel filter
- `mel_ends.hex` - end bin index per mel filter
- `log2_lut.hex` - 64-entry log₂ fractional LUT in Q4.12

> The `data/` directory is gitignored. Always run `mel_coeffs.py` after
> cloning before attempting to simulate any log-mel module.

**Step 2: Run individual module tests:**
```bash
cd src/rtl/Log-Mel/rtl/power_calc && make test-cocotb
cd src/rtl/Log-Mel/rtl/mac_unit && make test-cocotb
cd src/rtl/Log-Mel/rtl/output_buffer && make test-cocotb
```

**Step 3: Run top-level integration test:**
```bash
cd src/rtl/Log-Mel/rtl/log_top
make test-cocotb
```

Hex files are symlinked into the simulator working directory automatically, 
no manual setup needed beyond Step 1.

| Test | Module | What it verifies |
|------|--------|-----------------|
| `test_power_basic` | `power_calc` | Known-value re²+im² computation |
| `test_power_negative_inputs` | `power_calc` | Sign handling - squaring produces positive output |
| `test_power_golden` | `power_calc` | 200 random inputs vs numpy golden model |
| `test_mac_single` | `mac_unit` | Single multiply-accumulate |
| `test_mac_accumulate_multiple` | `mac_unit` | Multi-cycle accumulation correctness |
| `test_mac_clear` | `mac_unit` | Accumulator reset between frames |
| `test_mac_golden` | `mac_unit` | 20 random frames vs golden model |
| `test_load_and_drain` | `output_buffer` | Load 40 values, drain in order |
| `test_frame_sent_signal` | `output_buffer` | frame_sent_o pulses exactly once per frame |
| `test_backpressure_alternating` | `output_buffer` | Valid/ready handshake under backpressure |
| `test_two_consecutive_frames` | `output_buffer` | Clean reload between frames |
| `test_logmel_single_frame` | `logmel_top` | End-to-end frame vs torchaudio reference model |
| `test_logmel_two_frames` | `logmel_top` | Multi-frame correctness, MAC clear between frames |
| `test_logmel_cnn_backpressure` | `logmel_top` | End-to-end with random CNN backpressure |

---

### Notes

- Icarus Verilog produces warnings about constant selects in `always_*`
  processes from the PULP IP cores. These are known simulator limitations
  and do not affect correctness: all tests pass despite these warnings.
- The PULP `Log2` IP uses constructs unsupported by Icarus. `log_lut` uses
  a behavioral `` `ifndef SYNTHESIS `` block for simulation and the actual
  IP for synthesis.
- For ASIC synthesis, `$readmemh` initializations must be replaced with
  case-statement ROMs. See `scripts/mel_coeffs.py` for the planned
  case-statement output mode.

## Prerequisites

We use a custom fork of the [gf180mcuD PDK variant](https://github.com/wafer-space/gf180mcu) until all changes have been upstreamed.

To clone the latest PDK version, simply run `make clone-pdk`.

In the next step, install LibreLane by following the Nix-based installation instructions: https://librelane.readthedocs.io/en/latest/installation/nix_installation/index.html

## Implement the Design

This repository contains a Nix flake that provides a shell with the [`leo/gf180mcu`](https://github.com/librelane/librelane/tree/leo/gf180mcu) branch of LibreLane.

Simply run `nix-shell` in the root of this repository.

> [!NOTE]
> Since we are working on a branch of LibreLane, OpenROAD needs to be compiled locally. This will be done automatically by Nix, and the binary will be cached locally. 

With this shell enabled, run the implementation:

```
make librelane
```

## View the Design

After completion, you can view the design using the OpenROAD GUI:

```
make librelane-openroad
```

Or using KLayout:

```
make librelane-klayout
```

## Copying the Design to the Final Folder

To copy your latest run to the `final/` folder in the root directory of the repository, run the following command:

```
make copy-final
```

This will only work if the last run was completed without errors.

## Verification and Simulation

We use [cocotb](https://www.cocotb.org/), a Python-based testbench environment, for the verification of the chip.
The underlying simulator is Icarus Verilog (https://github.com/steveicarus/iverilog).

The testbench is located in `cocotb/chip_top_tb.py`. To run the RTL simulation, run the following command:

```
make sim
```

To run the GL (gate-level) simulation, run the following command:

```
make sim-gl
```

> [!NOTE]
> You need to have the latest implementation of your design in the `final/` folder. After implementing the design, execute 'make copy-final' to copy all necessary files.

In both cases, a waveform file will be generated under `cocotb/sim_build/chip_top.fst`.
You can view it using a waveform viewer, for example, [GTKWave](https://gtkwave.github.io/gtkwave/).

```
make sim-view
```

You can now update the testbench according to your design.

## Implementing Your Own Design

The source files for this template can be found in the `src/` directory. `chip_top.sv` defines the top-level ports and instantiates `chip_core`, chip ID (QR code) and the wafer.space logo. To allow for the default bonding setup, do not change the number of pads in order to keep the original bondpad positions. To be compatible with the default breakout PCB, do not change any of the power or ground pads. However, you can change the type of the signal pads, e.g. to bidirectional, input-only or e.g. analog pads. The template provides the `NUM_INPUT` and `NUM_BIDIR` parameters for this purpose.

The actual pad positions are defined in the LibreLane configuration file under `librelane/config.yaml`. The variables `PAD_SOUTH`/`PAD_EAST`/`PAD_NORTH`/`PAD_WEST` determine the respective pad placement. The LibreLane configuration also allows you to customize the flow (enable or disable steps), specify the source files, set various variables for the steps, and instantiate macros. For more information about the configuration, please refer to the LibreLane documentation: https://librelane.readthedocs.io/en/latest/

To implement your own design, simply edit `chip_core.sv`. The `chip_core` module receives the clock and reset, as well as the signals from the pads defined in `chip_top`. As an example, a 42-bit wide counter is implemented.

> [!NOTE]
> For more comprehensive SystemVerilog support, enable the `USE_SLANG` variable in the LibreLane configuration.

## Choosing a Different Slot Size

The template supports the following slot sizes: `1x1`, `0p5x1`, `1x0p5`, `0p5x0p5`.
By default, the design is implemented using the `1x1` slot definition.

To select a different slot size, simply set the `SLOT` environment variable.
This can be done when invoking a make target:

```
SLOT=0p5x0p5 make librelane
```

Alternatively, you can export the slot size:

```
export SLOT=0p5x0p5
```

You can change the slot that is selected by default in the Makefile by editing the value of `DEFAULT_SLOT`.

## Building a Standalone Padring for Analog Design

To build just the padring without any standard cell rows, digital routing or filler cells, run the following command:

```
make librelane-padring
```

It is also possible to build the padring for other slot sizes:

```
SLOT=0p5x0p5 make librelane-padring
```

## Precheck

To check whether your design is suitable for manufacturing, run the [gf180mcu-precheck](https://github.com/wafer-space/gf180mcu-precheck) with your layout.
