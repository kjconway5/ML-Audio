TOP ?= flash_top

SOURCES := \
	boot_pkg.sv \
	boot_controller.sv \
	features_boot_router.sv \
	dscnn_boot_router.sv \
	flash_top.sv

COCOTB_MODULE    ?= test_$(TOP)
COCOTB_MAKEFILES := $(shell cocotb-config --makefiles 2>/dev/null)

export COMPILE_ARGS += -DDUMP_WAVES -g2012

test-cocotb test-cocotb-icarus:
	SIM=icarus TOPLEVEL_LANG=verilog TOPLEVEL=$(TOP) MODULE=$(COCOTB_MODULE) \
		VERILOG_SOURCES="$(abspath $(SOURCES))" \
		$(MAKE) -f $(COCOTB_MAKEFILES)/Makefile.sim
.PHONY: test-cocotb test-cocotb-icarus

# Run the standalone boot_controller tests
test-boot-controller:
	$(MAKE) test-cocotb TOP=boot_controller
.PHONY: test-boot-controller

clean:
	rm -rf sim_build build waves __pycache__ .pytest_cache results.xml *.vcd *.fst *.log
.PHONY: clean
