TOP = subservient_test_tb

# Testbench and wrapper
SRCS = subservient_test_tb.sv \
       subservient_wrapper.sv \
       ../weight_sram/weight_sram.sv \
       ../../../../ip/subservient/rtl/subservient_core.v

# SRAM files
SRAMS = ../../../../gf180mcu_ocd_ip_sram/cells/gf180mcu_ocd_ip_sram__sram1024x8m8wm1/gf180mcu_ocd_ip_sram__sram1024x8m8wm1.v 

# SERV RTL files and subservient dependencies
SERV_SRCS = $(wildcard ../../../../ip/serv/servile/*.v) \
            $(wildcard ../../../../ip/serv/rtl/*.v)

all: test

test: fw
	iverilog -g2009 -DSIM -o sim.vvp $(SRCS) $(SRAMS) $(SERV_SRCS)
	vvp sim.vvp

fw:
	make -C fw all

clean:
	rm -f sim.vvp