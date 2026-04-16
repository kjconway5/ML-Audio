TOPLEVEL_LANG = verilog

VERILOG_SOURCES = \
    ../iCE40_Mic_Demo/top.sv \
    ../FIR/fastfir.v \
    ../FIR/firtap.v \
    ../CIC/cic.v

TOPLEVEL = top
MODULE = test_top

SIM = icarus

WAVES = 1

EXTRA_ARGS += -g2012

include $(shell cocotb-config --makefiles)/Makefile.sim