TOP      := feature_sram

SRAM_MACRO_DIR := $(abspath ../../../../gf180mcu_ocd_ip_sram/cells/gf180mcu_ocd_ip_sram__sram1024x8m8wm1)

SRAM_SOURCES := \
	$(SRAM_MACRO_DIR)/gf180mcu_ocd_ip_sram__sram1024x8m8wm1.v

SOURCES := \
	feature_sram.sv \
	$(SRAM_SOURCES)

COCOTB_MODULE    ?= test_$(TOP)
COCOTB_MAKEFILES := $(shell cocotb-config --makefiles 2>/dev/null)

test-cocotb test-cocotb-icarus:
	SIM=icarus TOPLEVEL_LANG=verilog TOPLEVEL=$(TOP) MODULE=$(COCOTB_MODULE) \
		VERILOG_SOURCES="$(abspath $(SOURCES))" \
		$(MAKE) -f $(COCOTB_MAKEFILES)/Makefile.sim
.PHONY: test-cocotb test-cocotb-icarus

test-cocotb-verilator:
	SIM=verilator TOPLEVEL_LANG=verilog TOPLEVEL=$(TOP) MODULE=$(COCOTB_MODULE) \
		VERILOG_SOURCES="$(abspath $(SOURCES))" \
		$(MAKE) -f $(COCOTB_MAKEFILES)/Makefile.sim
.PHONY: test-cocotb-verilator

clean:
	rm -rf sim_build build waves __pycache__ .pytest_cache results.xml *.vcd *.fst *.log
.PHONY: clean

help:
	@echo "feature_sram cocotb testbench"
	@echo ""
	@echo "Targets:"
	@echo "  test-cocotb            cocotb + icarus (default)"
	@echo "  test-cocotb-icarus     cocotb + icarus (explicit)"
	@echo "  test-cocotb-verilator  cocotb + verilator"
	@echo "  clean                  Remove build artefacts"
.PHONY: help
