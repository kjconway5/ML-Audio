# Simulation fragment
REPO_ROOT ?= $(shell git rev-parse --show-toplevel)
IVERILOG ?= iverilog
VVP ?= vvp
VERILATOR ?= verilator
GTKWAVE ?= gtkwave
PYTHON ?= python3

SIM_SOURCES = $(addprefix $(REPO_ROOT)/,$(shell $(PYTHON) $(REPO_ROOT)/util/get_filelist.py))
SIM_TOP = $(shell $(PYTHON) $(REPO_ROOT)/util/get_top.py)
TB_MODULE ?= tb_$(SIM_TOP)
RUN_DIR := run

# SystemVerilog simulation
test-sv: $(RUN_DIR)/$(TB_MODULE).vcd

$(RUN_DIR)/$(TB_MODULE).vcd: filelist.json $(SIM_SOURCES)
	@mkdir -p $(RUN_DIR)
	$(IVERILOG) -g2012 -o $(RUN_DIR)/$(TB_MODULE).vvp $(SIM_SOURCES)
	cd $(RUN_DIR) && $(VVP) $(TB_MODULE).vvp
	@[ -f $(TB_MODULE).vcd ] && mv $(TB_MODULE).vcd $(RUN_DIR)/ || true

wave-sv: $(RUN_DIR)/$(TB_MODULE).vcd
	$(GTKWAVE) $<

# Cocotb tests
test-cocotb:          ; pytest -rA --tb=short
test-cocotb-icarus:   ; COCOTB_SIMULATOR=icarus pytest -rA --tb=short
test-cocotb-verilator:; COCOTB_SIMULATOR=verilator pytest -rA --tb=short

wave-cocotb:
	@VCD=$$(find $(RUN_DIR) -name "*.vcd" -o -name "*.fst" 2>/dev/null | head -1); \
	[ -n "$$VCD" ] && $(GTKWAVE) $$VCD || echo "No waveform found"

# Linting
lint:
	$(VERILATOR) --lint-only -Wall -Wno-DECLFILENAME --top-module $(SIM_TOP) $(SIM_SOURCES)

# Cleanup
sim-clean:
	rm -rf $(RUN_DIR) build waves lint __pycache__ .pytest_cache *.vcd *.fst *.log

clean: sim-clean

# Help
help:
	@echo "Simulation: test-sv | wave-sv | test-cocotb | test-cocotb-icarus | test-cocotb-verilator | wave-cocotb"
	@echo "Linting:    lint"
	@echo "Cleanup:    sim-clean"

.PHONY: test-sv wave-sv test-cocotb test-cocotb-icarus test-cocotb-verilator wave-cocotb lint sim-clean clean help