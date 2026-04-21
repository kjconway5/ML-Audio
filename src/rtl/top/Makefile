TOP      := pipeline_top

STFFT_DIR    := $(abspath ../STFFT)
LOGMEL_DIR   := $(abspath ../Log-Mel)
RTL_DIR      := $(LOGMEL_DIR)/rtl
IP_DIR       := $(LOGMEL_DIR)/ip
SPECTBUF_DIR := $(abspath ../SPECT_BUFFER)


# R2FFT-based STFFT sources  (replaces old FFT_DIR block)
STFFT_SOURCES := \
 	$(STFFT_DIR)/R2FFT/hdl/bfp_Shifter.sv \
 	$(STFFT_DIR)/R2FFT/hdl/bfp_bitWidthAcc.sv \
 	$(STFFT_DIR)/R2FFT/hdl/bfp_bitWidthDetector.sv \
 	$(STFFT_DIR)/R2FFT/hdl/bfp_maxBitWidth.sv \
 	$(STFFT_DIR)/R2FFT/hdl/bitReverseCounter.sv \
 	$(STFFT_DIR)/R2FFT/hdl/ramPipelineBridge.sv \
 	$(STFFT_DIR)/R2FFT/hdl/radix2Butterfly.sv \
 	$(STFFT_DIR)/R2FFT/hdl/butterflyCore.sv \
 	$(STFFT_DIR)/R2FFT/hdl/twiddleFactorRomBridge.sv \
 	$(STFFT_DIR)/R2FFT/hdl/butterflyUnit.sv \
 	$(STFFT_DIR)/R2FFT/hdl/fftAddressGenerator.sv \
 	$(STFFT_DIR)/R2FFT/hdl/readBusMux.sv \
 	$(STFFT_DIR)/R2FFT/hdl/writeBusMux.sv \
 	$(STFFT_DIR)/R2FFT/hdl/R2FFT.sv \
 	$(STFFT_DIR)/tests/models/gf180mcu_fd_ip_sram__sram128x8m8wm1.v \
 	$(STFFT_DIR)/ZipCPU/ip/Window/windowfn.v \
 	$(STFFT_DIR)/rtl/fft_data_ram.sv \
 	$(STFFT_DIR)/rtl/fft_twiddle_rom.sv \
 	$(STFFT_DIR)/rtl/stfft.sv

LOGMEL_SOURCES := \
	$(RTL_DIR)/log_top/logmel_top.sv \
	$(RTL_DIR)/power_calc/power_calc.sv \
	$(RTL_DIR)/mel_filterbank/mel_filterbank.sv \
	$(RTL_DIR)/mel_filterbank/mel_coeff_sram.sv \
	$(RTL_DIR)/mac_unit/mac_unit.sv \
	$(RTL_DIR)/frame_control/frame_control.sv \
	$(RTL_DIR)/log_lut/log_lut.sv \
	$(RTL_DIR)/log_lut/log_lut_sram.sv \
	$(RTL_DIR)/output_buffer/output_buffer.sv

IP_SOURCES := $(wildcard $(IP_DIR)/*.sv)

SPECTBUF_SOURCES := \
	$(SPECTBUF_DIR)/spect_buffer_ctrl.sv

SOURCES := pipeline_top.sv \
	$(STFFT_SOURCES) \
	$(LOGMEL_SOURCES) \
	$(IP_SOURCES) \
	$(SPECTBUF_SOURCES)

COCOTB_MODULE    ?= test_$(TOP)
COCOTB_MAKEFILES := $(shell cocotb-config --makefiles 2>/dev/null)


# Hex data files required by $readmemh in RTL
DATA_DIR    := $(abspath $(LOGMEL_DIR)/data)
HANNING_HEX := $(STFFT_DIR)/ZipCPU/hanning.hex

HEX_FILES := \
	$(DATA_DIR)/mel_coeffs.hex \
	$(DATA_DIR)/mel_starts.hex \
	$(DATA_DIR)/mel_ends.hex \
	$(DATA_DIR)/mel_coeffs_sparse.hex \
	$(DATA_DIR)/mel_indices.hex \
	$(DATA_DIR)/log2_lut.hex



# R2FFT twiddle ROM hex files (fft_twiddle_rom.sv reads these)
TWIDDLE_HEX_FILES := $(wildcard $(STFFT_DIR)/rtl/*.hex)

link-hex:
	@ln -sf $(HANNING_HEX)             . 2>/dev/null || true
	@for f in $(HEX_FILES);            do ln -sf $$f . 2>/dev/null || true; done
	@for f in $(CMEM_HEX_FILES);       do ln -sf $$f . 2>/dev/null || true; done
	@ln -sfn $(DATA_DIR) data 2>/dev/null || true
# R2FFT twiddle hex (uncomment when switching to R2FFT):
	@for f in $(TWIDDLE_HEX_FILES);    do ln -sf $$f . 2>/dev/null || true; done
.PHONY: link-hex


# Simulation targets
test-cocotb test-cocotb-icarus: link-hex
	SIM=icarus TOPLEVEL_LANG=verilog TOPLEVEL=$(TOP) MODULE=$(COCOTB_MODULE) \
		VERILOG_SOURCES="$(abspath $(SOURCES))" \
		$(MAKE) -f $(COCOTB_MAKEFILES)/Makefile.sim
.PHONY: test-cocotb test-cocotb-icarus

test-cocotb-verilator: link-hex
	SIM=verilator TOPLEVEL_LANG=verilog TOPLEVEL=$(TOP) MODULE=$(COCOTB_MODULE) \
		VERILOG_SOURCES="$(abspath $(SOURCES))" \
		EXTRA_ARGS="--trace-fst -Wno-fatal" \
		$(MAKE) -f $(COCOTB_MAKEFILES)/Makefile.sim
.PHONY: test-cocotb-verilator

# ---------------------------------------------------------------------------

clean:
	rm -rf sim_build build waves __pycache__ .pytest_cache results.xml \
		*.vcd *.fst *.log \
		hanning.hex mel_coeffs_sparse.hex mel_indices.hex log2_lut.hex \
		mel_coeffs.hex mel_starts.hex mel_ends.hex \
		cmem_*.hex twiddle_*.hex \
		rtl_features.npy
.PHONY: clean

help:
	@echo "pipeline_top cocotb testbench (STFFT + Log-Mel end-to-end pipeline)"
	@echo ""
	@echo "Targets:"
	@echo "  test-cocotb            cocotb + icarus (default)"
	@echo "  test-cocotb-icarus     cocotb + icarus (explicit)"
	@echo "  test-cocotb-verilator  cocotb + verilator"
	@echo "  clean                  Remove build artefacts and hex symlinks"
	@echo ""
	@echo "Optional variables:"
	@echo "  WAV_FILE=/path/to/audio.wav  Feed a real WAV file instead of the"
	@echo "                               synthetic chirp test signal."
.PHONY: help