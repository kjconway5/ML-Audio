TOP      := pipeline_top

STFFT_DIR  := $(abspath ../STFFT)
LOGMEL_DIR := $(abspath ../Log-Mel)
RTL_DIR    := $(LOGMEL_DIR)/rtl
IP_DIR     := $(LOGMEL_DIR)/ip

FFT_DIR    := $(STFFT_DIR)/ip/FFT

STFFT_SOURCES := \
	$(STFFT_DIR)/stfft.sv \
	$(FFT_DIR)/fftmain.v \
	$(FFT_DIR)/fftstage.v \
	$(FFT_DIR)/butterfly.v \
	$(FFT_DIR)/hwbfly.v \
	$(FFT_DIR)/qtrstage.v \
	$(FFT_DIR)/laststage.v \
	$(FFT_DIR)/bitreverse.v \
	$(FFT_DIR)/bimpy.v \
	$(FFT_DIR)/longbimpy.v \
	$(FFT_DIR)/convround.v \
	$(FFT_DIR)/windowfn.v

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

SOURCES := pipeline_top.sv $(STFFT_SOURCES) $(LOGMEL_SOURCES) $(IP_SOURCES)

COCOTB_MODULE    ?= test_$(TOP)
COCOTB_MAKEFILES := $(shell cocotb-config --makefiles 2>/dev/null)

# Hex data files required by $readmemh in RTL
DATA_DIR    := $(abspath $(LOGMEL_DIR)/data)
HANNING_HEX := $(FFT_DIR)/hanning.hex
HEX_FILES   := \
	$(DATA_DIR)/mel_coeffs.hex \
	$(DATA_DIR)/mel_starts.hex \
	$(DATA_DIR)/mel_ends.hex \
	$(DATA_DIR)/log2_lut.hex

CMEM_HEX_FILES := $(wildcard $(FFT_DIR)/cmem_*.hex)

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
	@echo "pipeline_top cocotb testbench (STFFT + Log-Mel end-to-end pipeline)"
	@echo ""
	@echo "Targets:"
	@echo "  test-cocotb            cocotb + icarus (default)"
	@echo "  test-cocotb-icarus     cocotb + icarus (explicit)"
	@echo "  test-cocotb-verilator  cocotb + verilator"
	@echo "  clean                  Remove build artefacts and hex symlinks"
.PHONY: help
