TOP    := compFIR
RTL    := $(abspath ../rtl)

SOURCES := $(RTL)/compFIR.sv

COCOTB_MODULE    ?= test_compFIR
COCOTB_MAKEFILES := $(shell cocotb-config --makefiles 2>/dev/null)

# ---------------------------------------------------------------------------
# Simulation targets
# ---------------------------------------------------------------------------

test-cocotb test-cocotb-verilator:
	SIM=verilator TOPLEVEL_LANG=verilog TOPLEVEL=$(TOP) MODULE=$(COCOTB_MODULE) \
		VERILOG_SOURCES="$(abspath $(SOURCES))" \
		$(MAKE) -f $(COCOTB_MAKEFILES)/Makefile.sim
.PHONY: test-cocotb test-cocotb-verilator

test-cocotb-icarus:
	SIM=icarus TOPLEVEL_LANG=verilog TOPLEVEL=$(TOP) MODULE=$(COCOTB_MODULE) \
		VERILOG_SOURCES="$(abspath $(SOURCES))" \
		$(MAKE) -f $(COCOTB_MAKEFILES)/Makefile.sim
.PHONY: test-cocotb-icarus

# Run the continuous test script directly (no cocotb, pure Python + numpy)
test-cont:
	python3 contfest.py
.PHONY: test-cont

# ---------------------------------------------------------------------------

clean:
	rm -rf sim_build build waves __pycache__ .pytest_cache results.xml \
		*.vcd *.fst *.log
.PHONY: clean

help:
	@echo "compFIR cocotb testbench"
	@echo ""
	@echo "Targets:"
	@echo "  test-cocotb           Verilator simulation (default)"
	@echo "  test-cocotb-icarus    Icarus simulation"
	@echo "  test-cont             Pure-Python frequency response test (contfest.py)"
	@echo "  clean                 Remove build artefacts"
.PHONY: help