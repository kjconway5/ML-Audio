TOP    := compFIR
RTL    := $(abspath ../rtl)

SOURCES := $(RTL)/compFIR.sv

COCOTB_MODULE    ?= test_compFIR
COCOTB_MAKEFILES := $(shell cocotb-config --makefiles 2>/dev/null)



test-FIR test-FIR-verilator:
	SIM=verilator TOPLEVEL_LANG=verilog TOPLEVEL=$(TOP) MODULE=$(COCOTB_MODULE) \
		VERILOG_SOURCES="$(abspath $(SOURCES))" \
		$(MAKE) -f $(COCOTB_MAKEFILES)/Makefile.sim
.PHONY: test-cocotb test-cocotb-verilator

test-FIR-icarus:
	SIM=icarus TOPLEVEL_LANG=verilog TOPLEVEL=$(TOP) MODULE=$(COCOTB_MODULE) \
		VERILOG_SOURCES="$(abspath $(SOURCES))" \
		$(MAKE) -f $(COCOTB_MAKEFILES)/Makefile.sim
.PHONY: test-cocotb-icarus


clean:
	rm -rf sim_build build waves __pycache__ .pytest_cache results.xml \
		*.vcd *.fst *.log
.PHONY: clean

help:
	@echo "compFIR cocotb testbench"
	@echo ""
	@echo "Targets:"
	@echo "  test-FIR           Verilator simulation (default)"
	@echo "  test-FIR-icarus    Icarus simulation"
	@echo "  clean                 Remove build artefacts"
.PHONY: help
