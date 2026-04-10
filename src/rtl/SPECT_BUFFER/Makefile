TOP        := spect_buffer_ctrl
SOURCES    := spect_buffer_ctrl.sv
TB_SOURCES := tb_$(TOP).sv

TB_MODULE  := tb_$(TOP)
RUN_DIR    := run

IVERILOG   ?= iverilog
VVP        ?= vvp
VERILATOR  ?= verilator
GTKWAVE    ?= gtkwave

ALL_SOURCES := $(SOURCES) $(TB_SOURCES)

.DEFAULT_GOAL := help

lint:
	$(VERILATOR) --lint-only -Wall -Wno-DECLFILENAME \
		--top-module $(TOP) $(SOURCES)
.PHONY: lint

test: $(RUN_DIR)/$(TB_MODULE).vcd

$(RUN_DIR)/$(TB_MODULE).vcd: $(ALL_SOURCES)
	@mkdir -p $(RUN_DIR)
	$(IVERILOG) -g2012 -o $(RUN_DIR)/$(TB_MODULE).vvp $(ALL_SOURCES)
	cd $(RUN_DIR) && $(VVP) $(TB_MODULE).vvp
	@[ -f $(TB_MODULE).vcd ] && mv $(TB_MODULE).vcd $(RUN_DIR)/ || true
.PHONY: test

# Waveform viewer
wave: $(RUN_DIR)/$(TB_MODULE).vcd
	$(GTKWAVE) $<
.PHONY: wave

# Cleanup
clean-local:
	rm -rf $(RUN_DIR) sim_build build waves __pycache__ .pytest_cache results.xml *.vcd *.fst *.log
.PHONY: clean-local

# Help
help:
	@echo "RTL Development Makefile"
	@echo ""
	@echo "  Current target:  $(TOP)"
	@echo "  Sources:         $(SOURCES)"
	@echo "  Testbench:       $(TB_SOURCES)"
	@echo ""
	@echo "Targets:"
	@echo "  lint          Verilator lint (design only)"
	@echo "  test          iverilog simulation"
	@echo "  wave          Open waveform in GTKWave"
	@echo "  clean-local   Remove local non-cocotb build artifacts"
	@echo "  sim           cocotb run"
	@echo "  regression    cocotb regression run"
.PHONY: help

# -------------------------
# cocotb configuration
# -------------------------
SIM             ?= icarus
TOPLEVEL_LANG   ?= verilog
WAVES           ?= 1

TOPLEVEL        := $(TOP)
MODULE          := test_$(TOP)
VERILOG_SOURCES := $(abspath $(SOURCES))

include $(shell cocotb-config --makefiles)/Makefile.sim