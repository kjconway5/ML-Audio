TOP      := pipeline_top

STFFT_DIR  := $(abspath ../STFFT)
LOGMEL_DIR := $(abspath ../Log-Mel)
RTL_DIR    := $(LOGMEL_DIR)/rtl
IP_DIR     := $(LOGMEL_DIR)/ip

STFFT_SOURCES := \
	$(STFFT_DIR)/stfft.sv \
	$(STFFT_DIR)/FFT/fftmain.v \
	$(STFFT_DIR)/FFT/fftstage.v \
	$(STFFT_DIR)/FFT/butterfly.v \
	$(STFFT_DIR)/FFT/hwbfly.v \
	$(STFFT_DIR)/FFT/qtrstage.v \
	$(STFFT_DIR)/FFT/laststage.v \
	$(STFFT_DIR)/FFT/bitreverse.v \
	$(STFFT_DIR)/FFT/bimpy.v \
	$(STFFT_DIR)/FFT/longbimpy.v \
	$(STFFT_DIR)/FFT/convround.v \
	$(STFFT_DIR)/FFT/windowfn.v

LOGMEL_SOURCES := \
	$(RTL_DIR)/log_top/logmel_top.sv \
	$(RTL_DIR)/power_calc/power_calc.sv \
	$(RTL_DIR)/mel_filterbank/mel_filterbank.sv \
	$(RTL_DIR)/mel_coeff_rom.sv \
	$(RTL_DIR)/mac_unit/mac_unit.sv \
	$(RTL_DIR)/frame_control/frame_control.sv \
	$(RTL_DIR)/log_lut/log_lut.sv \
	$(RTL_DIR)/output_buffer/output_buffer.sv

IP_SOURCES := $(wildcard $(IP_DIR)/*.sv)

SOURCES := features_top.sv $(STFFT_SOURCES) $(LOGMEL_SOURCES) $(IP_SOURCES)

COCOTB_MODULE    ?= test_$(TOP)
COCOTB_MAKEFILES := $(shell cocotb-config --makefiles 2>/dev/null)

# Hex data files required by $readmemh in RTL
DATA_DIR    := $(abspath $(LOGMEL_DIR)/data)
HANNING_HEX := $(STFFT_DIR)/hanning.hex
HEX_FILES   := \
	$(DATA_DIR)/mel_coeffs.hex \
	$(DATA_DIR)/mel_starts.hex \
	$(DATA_DIR)/mel_ends.hex \
	$(DATA_DIR)/log2_lut.hex

CMEM_HEX_FILES := $(wildcard $(STFFT_DIR)/FFT/cmem_*.hex)

link-hex:
	@ln -sf $(HANNING_HEX) . 2>/dev/null || true
	@for f in $(HEX_FILES); do ln -sf $$f . 2>/dev/null || true; done
	@for f in $(CMEM_HEX_FILES); do ln -sf $$f . 2>/dev/null || true; done
.PHONY: link-hex

test-cocotb test-cocotb-icarus: link-hex
	SIM=icarus TOPLEVEL_LANG=verilog TOPLEVEL=$(TOP) MODULE=$(COCOTB_MODULE) \
		VERILOG_SOURCES="$(abspath $(SOURCES))" \
		$(MAKE) -f $(COCOTB_MAKEFILES)/Makefile.sim
.PHONY: test-cocotb test-cocotb-icarus

test-cocotb-verilator: link-hex
	SIM=verilator TOPLEVEL_LANG=verilog TOPLEVEL=$(TOP) MODULE=$(COCOTB_MODULE) \
		VERILOG_SOURCES="$(abspath $(SOURCES))" \
		$(MAKE) -f $(COCOTB_MAKEFILES)/Makefile.sim
.PHONY: test-cocotb-verilator

clean:
	rm -rf sim_build build waves __pycache__ .pytest_cache results.xml *.vcd *.fst *.log \
		hanning.hex mel_coeffs.hex mel_starts.hex mel_ends.hex log2_lut.hex cmem_*.hex
.PHONY: clean

help:
	@echo "features_top cocotb testbench (STFFT + Log-Mel end-to-end pipeline)"
	@echo ""
	@echo "Targets:"
	@echo "  test-cocotb            cocotb + icarus (default)"
	@echo "  test-cocotb-icarus     cocotb + icarus (explicit)"
	@echo "  test-cocotb-verilator  cocotb + verilator"
	@echo "  clean                  Remove build artefacts and hex symlinks"
	@echo ""
	@echo "Optional variables:"
	@echo "  WAV_FILE=/path/to/audio.wav  Feed a real WAV file instead of the"
	@echo "                               synthetic multi-tone test signal."
	@echo "  Example:"
	@echo "    WAV_FILE=~/yes.wav make test-cocotb"
	@echo ""
	@echo "Output:"
	@echo "  spectrogram_comparison.png — three-panel plot saved in this directory"
.PHONY: help