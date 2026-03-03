

TOP        := logmel_top
SOURCES    := logmel_top.sv \
              ../power_calc/power_calc.sv \
              ../mel_filterbank/mel_filterbank.sv \
              ../mel_coeff_rom.sv \
              ../mac_unit/mac_unit.sv \
              ../frame_control/frame_control.sv \
              ../log_lut/log_lut.sv \
              ../output_buffer/output_buffer.sv \
              $(wildcard ../../ip/*.sv)
TB_SOURCES :=

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

# Cocotb tests
COCOTB_MODULE ?= test_$(TOP)
COCOTB_MAKEFILES := $(shell cocotb-config --makefiles 2>/dev/null)

DATA_DIR   := $(abspath ../../data)
HEX_FILES  := $(wildcard $(DATA_DIR)/*.hex)

link-hex:
	@for f in $(HEX_FILES); do ln -sf $$f . 2>/dev/null || true; done
.PHONY: link-hex

test-cocotb test-cocotb-icarus: link-hex
	SIM=icarus TOPLEVEL_LANG=verilog TOPLEVEL=$(TOP) MODULE=$(COCOTB_MODULE) \
		VERILOG_SOURCES="$(abspath $(SOURCES))" \
		$(MAKE) -f $(COCOTB_MAKEFILES)/Makefile.sim
test-cocotb-verilator: link-hex
	SIM=verilator TOPLEVEL_LANG=verilog TOPLEVEL=$(TOP) MODULE=$(COCOTB_MODULE) \
		VERILOG_SOURCES="$(abspath $(SOURCES))" \
		$(MAKE) -f $(COCOTB_MAKEFILES)/Makefile.sim
.PHONY: test-cocotb test-cocotb-icarus test-cocotb-verilator

# Waveform viewer
wave: $(RUN_DIR)/$(TB_MODULE).vcd
	$(GTKWAVE) $<
.PHONY: wave

# Cleanup
clean:
	rm -rf $(RUN_DIR) sim_build build waves __pycache__ .pytest_cache results.xml *.vcd *.fst *.log
.PHONY: clean

# Help
help:
	@echo "RTL Development Makefile"
	@echo ""
	@echo "  Current target:  $(TOP)"
	@echo "  Sources:         $(SOURCES)"
	@echo "  Testbench:       $(TB_SOURCES)"
	@echo ""
	@echo "Targets:"
	@echo "  lint                 Verilator lint (design only)"
	@echo "  test                 iverilog simulation"
	@echo "  test-cocotb          cocotb + pytest (default simulator)"
	@echo "  test-cocotb-icarus   cocotb + pytest (icarus)"
	@echo "  test-cocotb-verilator cocotb + pytest (verilator)"
	@echo "  wave                 Open waveform in GTKWave"
	@echo "  clean                Remove build artifacts"
