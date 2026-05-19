TOP     := fft
FFT_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)

SOURCES := \
    $(FFT_ROOT)/hdl/bfp_Shifter.sv \
    $(FFT_ROOT)/hdl/bfp_bitWidthAcc.sv \
    $(FFT_ROOT)/hdl/bfp_bitWidthDetector.sv \
    $(FFT_ROOT)/hdl/bfp_maxBitWidth.sv \
    $(FFT_ROOT)/hdl/bitReverseCounter.sv \
    $(FFT_ROOT)/hdl/rwBusMux.sv \
    $(FFT_ROOT)/hdl/radix2Butterfly.sv \
    $(FFT_ROOT)/hdl/butterflyCore.sv \
    $(FFT_ROOT)/hdl/twiddleFactorRomBridge.sv \
    $(FFT_ROOT)/hdl/butterflyUnit.sv \
    $(FFT_ROOT)/hdl/fftAddressGenerator.sv \
    $(FFT_ROOT)/hdl/R2FFT.sv \
    $(FFT_ROOT)/hdl/fft_twiddle_rom.sv \
	$(FFT_ROOT)/hdl/ramPipelineBridge.sv \
    $(FFT_ROOT)/hdl/fft.sv

COCOTB_MAKEFILES := $(shell cocotb-config --makefiles)

# Waveform generation (use VCD for better compatibility)
SIM_ARGS = --trace --trace-structs --trace-depth 2

export COCOTB_HDL_TIMEUNIT = 1ns
export COCOTB_HDL_TIMEPRECISION = 1ps

test-fft:
	@echo "Creating hanning.hex symlink..."
	ln -sf $(STFFT_ROOT)/ZipCPU/hanning.hex hanning.hex
	@echo "Running simulation with waveform generation..."
	SIM=verilator \
	TOPLEVEL_LANG=verilog \
	TOPLEVEL=$(TOP) \
	MODULE=test_$(TOP) \
	COCOTB_TEST_MODULES=test_$(TOP) \
	EXTRA_ARGS="-Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -DSIM -Wno-fatal $(SIM_ARGS)" \
	VERILOG_SOURCES="$(SOURCES)" \
	$(MAKE) -f $(COCOTB_MAKEFILES)/Makefile.sim

test-debug:
	@echo "Creating hanning.hex symlink..."
	ln -sf $(STFFT_ROOT)/ZipCPU/hanning.hex hanning.hex
	@echo "Running debug test..."
	SIM=verilator \
	TOPLEVEL_LANG=verilog \
	TOPLEVEL=$(TOP) \
	MODULE=test_$(TOP) \
	COCOTB_TEST_MODULES=test_$(TOP) \
	EXTRA_ARGS="-Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-fatal $(SIM_ARGS)" \
	VERILOG_SOURCES="$(SOURCES)" \
	$(MAKE) -f $(COCOTB_MAKEFILES)/Makefile.sim TESTCASE=test_debug_window

clean:
	rm -rf sim_build build waves __pycache__ .pytest_cache results.xml *.vcd *.fst *.log hanning.hex

.PHONY: test-stfft test-debug clean