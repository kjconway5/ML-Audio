TOP := kws_top

REPO_ROOT      := $(abspath ../../..)
DSCNN_DIR      := $(abspath ..)
SRAM_MACRO_DIR := $(abspath $(REPO_ROOT)/../gf180mcu_ocd_ip_sram/cells/gf180mcu_ocd_ip_sram__sram1024x8m8wm1)

SRAM_MACRO := $(SRAM_MACRO_DIR)/gf180mcu_ocd_ip_sram__sram1024x8m8wm1.v

SOURCES := \
	$(DSCNN_DIR)/kws_top.sv \
	$(DSCNN_DIR)/fsm/FSM.sv \
	$(DSCNN_DIR)/spectrogram_sram/spectrogram_sram.sv \
	$(DSCNN_DIR)/weight_sram/weight_sram.sv \
	$(DSCNN_DIR)/feature_sram/feature_sram.sv \
	$(DSCNN_DIR)/mac_array/mac_array.sv \
	$(DSCNN_DIR)/requant/requant.sv \
	$(DSCNN_DIR)/bias_dff/bias_DFFs.sv \
	$(SRAM_MACRO)

# Simulation sources: exclude the GF180 SRAM macro (replaced by behavioral models)
SIM_SOURCES := \
	$(DSCNN_DIR)/kws_top.sv \
	$(DSCNN_DIR)/fsm/FSM.sv \
	$(DSCNN_DIR)/spectrogram_sram/spectrogram_sram.sv \
	$(DSCNN_DIR)/weight_sram/weight_sram.sv \
	$(DSCNN_DIR)/feature_sram/feature_sram.sv \
	$(DSCNN_DIR)/mac_array/mac_array.sv \
	$(DSCNN_DIR)/requant/requant.sv \
	$(DSCNN_DIR)/bias_dff/bias_DFFs.sv

COCOTB_MODULE    ?= test_$(TOP)
COCOTB_MAKEFILES := $(shell cocotb-config --makefiles 2>/dev/null)

# Generate test vectors (run once before simulation)
gen-spect:
	python3 $(abspath generate_spect.py) \
		--keyword yes \
		--out-dir $(abspath .)
.PHONY: gen-spect

test-cocotb test-cocotb-icarus:
	SIM=icarus TOPLEVEL_LANG=verilog TOPLEVEL=$(TOP) COCOTB_TEST_MODULES=$(COCOTB_MODULE) \
		VERILOG_SOURCES="$(abspath $(SIM_SOURCES))" \
		COMPILE_ARGS="-DSIM" \
		$(MAKE) -f $(COCOTB_MAKEFILES)/Makefile.sim
.PHONY: test-cocotb test-cocotb-icarus

clean:
	rm -rf sim_build build waves __pycache__ .pytest_cache results.xml *.vcd *.fst *.log
.PHONY: clean

help:
	@echo "kws_top end-to-end inference testbench"
	@echo ""
	@echo "  Step 1 — generate test vectors (run once, or after retraining):"
	@echo "    make gen-spect          (requires dataset at src/ml/Pipeline/speech_commands_v0.01)"
	@echo ""
	@echo "  Step 2 — run the simulation:"
	@echo "    make test-cocotb"
	@echo ""
	@echo "Targets:"
	@echo "  gen-spect    Generate spectrogram.hex + expected_class.txt from dataset"
	@echo "  test-cocotb    Run full inference simulation with icarus + cocotb"
	@echo "  clean          Remove build artefacts"
.PHONY: help
