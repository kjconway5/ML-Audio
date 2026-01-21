# Fragment makefile for simulation targets
# Supports both SystemVerilog testbenches (Icarus/Verilator) and Cocotb Python tests

# Path to the repository root
REPO_ROOT ?= $(shell git rev-parse --show-toplevel)

# Tool paths (override if installed in non-standard locations)
IVERILOG ?= iverilog
VVP ?= vvp
VERILATOR ?= verilator
GTKWAVE ?= gtkwave
PYTHON ?= python3

# Read source files and top module from filelist.json
SIM_SOURCES = $(shell $(PYTHON) $(REPO_ROOT)/util/get_filelist.py)
SIM_SOURCES := $(addprefix $(REPO_ROOT)/,$(SIM_SOURCES))
SIM_TOP = $(shell $(PYTHON) $(REPO_ROOT)/util/get_top.py)

# Default testbench name (override in module Makefile if different)
TB_MODULE ?= tb_$(SIM_TOP)

# Output directories
RUN_DIR := run
BUILD_DIR := build
WAVE_DIR := waves

all: help

#==============================================================================
# SystemVerilog Testbench Targets (Icarus Verilog)
#==============================================================================

# Compile and run SystemVerilog testbench with Icarus
test-sv: $(RUN_DIR)/$(TB_MODULE).vcd
	@echo "SystemVerilog simulation complete. Waveform: $(RUN_DIR)/$(TB_MODULE).vcd"

$(RUN_DIR)/$(TB_MODULE).vcd: filelist.json $(SIM_SOURCES)
	@mkdir -p $(RUN_DIR)
	@echo "Compiling with Icarus Verilog..."
	$(IVERILOG) -g2012 -o $(RUN_DIR)/$(TB_MODULE).vvp $(SIM_SOURCES) 2>&1 | tee $(RUN_DIR)/compile.log
	@echo "Running simulation..."
	cd $(RUN_DIR) && $(VVP) $(TB_MODULE).vvp | tee sim.log
	@if [ -f $(TB_MODULE).vcd ]; then mv $(TB_MODULE).vcd $(RUN_DIR)/; fi

# View waveform from SystemVerilog testbench
wave-sv: $(RUN_DIR)/$(TB_MODULE).vcd
	$(GTKWAVE) $(RUN_DIR)/$(TB_MODULE).vcd

#==============================================================================
# Cocotb Testbench Targets (Python)
#==============================================================================

# Run Cocotb tests with pytest
test-cocotb: filelist.json $(SIM_SOURCES)
	@echo "Running Cocotb tests..."
	pytest -rA --tb=short

test-cocotb-icarus: 
	@echo "Running Cocotb tests with Icarus Verilog..."
	COCOTB_SIMULATOR=icarus pytest -rA --tb=short

test-cocotb-verilator:
	@echo "Running Cocotb tests with Verilator..."
	COCOTB_SIMULATOR=verilator pytest -rA --tb=short

# View waveform from Cocotb test (assuming VCD output)
wave-cocotb:
	@if [ -d $(RUN_DIR) ]; then \
		VCD_FILE=$$(find $(RUN_DIR) -name "*.vcd" -o -name "*.fst" | head -n 1); \
		if [ -n "$$VCD_FILE" ]; then \
			echo "Opening waveform: $$VCD_FILE"; \
			$(GTKWAVE) $$VCD_FILE; \
		else \
			echo "No waveform files found in $(RUN_DIR)"; \
		fi \
	else \
		echo "No run directory found. Run 'make test-cocotb' first."; \
	fi

#==============================================================================
# Linting and Verification
#==============================================================================

# Lint all source files with Verilator
lint:
	@echo "Running Verilator lint..."
	$(VERILATOR) --lint-only -Wall -Wno-DECLFILENAME \
		--top-module $(SIM_TOP) $(SIM_SOURCES) 2>&1 | tee lint.log

# Style check (can be extended)
style:
	@echo "Running style checks..."
	@echo "Style checking not yet implemented"

#==============================================================================
# Cleanup Targets
#==============================================================================

sim-clean:
	rm -rf $(RUN_DIR)
	rm -rf $(BUILD_DIR)
	rm -rf $(WAVE_DIR)
	rm -rf lint
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf *.vcd *.fst
	rm -rf *.log

sim-extraclean: sim-clean
	rm -f .coverage

#==============================================================================
# Help Targets
#==============================================================================

sim-help:
	@echo ""
	@echo "Simulation Targets:"
	@echo "  test-sv            : Compile and run SystemVerilog testbench with Icarus"
	@echo "  wave-sv            : View waveform from SystemVerilog simulation"
	@echo "  test-cocotb        : Run all Cocotb tests"
	@echo "  test-cocotb-icarus : Run Cocotb tests with Icarus Verilog"
	@echo "  test-cocotb-verilator : Run Cocotb tests with Verilator"
	@echo "  wave-cocotb        : View waveform from Cocotb simulation"
	@echo "  lint               : Run Verilator lint on all sources"
	@echo "  style              : Run style checks"
	@echo "  sim-clean          : Remove all simulation outputs"
	@echo "  sim-extraclean     : Remove all generated files (includes sim-clean)"

sim-vars-help:
	@echo ""
	@echo "Simulation Variables (override in environment or command line):"
	@echo "  IVERILOG    : Path to iverilog executable"
	@echo "  VVP         : Path to vvp executable"
	@echo "  VERILATOR   : Path to verilator executable"
	@echo "  GTKWAVE     : Path to gtkwave executable"
	@echo "  PYTHON      : Path to python3 executable"
	@echo "  TB_MODULE   : Testbench module name (default: tb_<top>)"

clean: sim-clean
targets-help: sim-help
vars-help: sim-vars-help

.PHONY: all test-sv wave-sv test-cocotb test-cocotb-icarus test-cocotb-verilator wave-cocotb \
        lint style sim-clean sim-extraclean sim-help sim-vars-help \
        clean targets-help vars-help