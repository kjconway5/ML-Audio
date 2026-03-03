
TOP     := logmel_top
RTL_DIR := ..
IP_DIR  := ../../ip

RTL_SOURCES := \
	logmel_top.sv \
	$(RTL_DIR)/power_calc/power_calc.sv \
	$(RTL_DIR)/mel_filterbank/mel_filterbank.sv \
	$(RTL_DIR)/mel_coeff_rom.sv \
	$(RTL_DIR)/mac_unit/mac_unit.sv \
	$(RTL_DIR)/frame_control/frame_control.sv \
	$(RTL_DIR)/log_lut/log_lut.sv \
	$(RTL_DIR)/output_buffer/output_buffer.sv

IP_SOURCES := $(wildcard $(IP_DIR)/*.sv)

SOURCES := $(RTL_SOURCES) $(IP_SOURCES)

COCOTB_MODULE    ?= test_$(TOP)
COCOTB_MAKEFILES := $(shell cocotb-config --makefiles 2>/dev/null)

# Hex data files required by $readmemh in RTL
DATA_DIR  := $(abspath ../../data)
HEX_FILES := \
	$(DATA_DIR)/mel_coeffs.hex \
	$(DATA_DIR)/mel_starts.hex \
	$(DATA_DIR)/mel_ends.hex \
	$(DATA_DIR)/log2_lut.hex

link-hex:
	@for f in $(HEX_FILES); do ln -sf $$f . 2>/dev/null || true; done
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
		mel_coeffs.hex mel_starts.hex mel_ends.hex log2_lut.hex
.PHONY: clean

help:
	@echo "logmel_top top-level cocotb testbench"
	@echo ""
	@echo "Targets:"
	@echo "  test-cocotb          cocotb + icarus (default)"
	@echo "  test-cocotb-icarus   cocotb + icarus (explicit)"
	@echo "  test-cocotb-verilator cocotb + verilator"
	@echo "  clean                Remove build artefacts and hex symlinks"
.PHONY: help
