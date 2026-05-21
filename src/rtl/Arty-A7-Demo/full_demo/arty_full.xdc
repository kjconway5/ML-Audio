# arty_full.xdc — Stage 3 pin map for Arty A7-100t. Same pins as
# features_stage2; separate file in case Stage 4/5 wants to add I2S
# mic without disturbing this one.

set_property -dict { PACKAGE_PIN E3  IOSTANDARD LVCMOS33 } [get_ports CLK100MHZ]
create_clock -name sys_clk_pin -period 10.000 [get_ports CLK100MHZ]

set_property -dict { PACKAGE_PIN D9  IOSTANDARD LVCMOS33 } [get_ports btn0]

set_property -dict { PACKAGE_PIN D10 IOSTANDARD LVCMOS33 } [get_ports uart_rxd_out]
set_property -dict { PACKAGE_PIN A9  IOSTANDARD LVCMOS33 } [get_ports uart_txd_in]

# LED pin map matches the Arty A7 silkscreen / Digilent master XDC.
# led4..led7 = monochrome green LEDs at the board edge.
# led0..led3 = green channel of the 4 RGB LEDs (R/B left floating).
set_property -dict { PACKAGE_PIN F6  IOSTANDARD LVCMOS33 } [get_ports led0]   ;# LD0 green
set_property -dict { PACKAGE_PIN J4  IOSTANDARD LVCMOS33 } [get_ports led1]   ;# LD1 green
set_property -dict { PACKAGE_PIN J2  IOSTANDARD LVCMOS33 } [get_ports led2]   ;# LD2 green
set_property -dict { PACKAGE_PIN H6  IOSTANDARD LVCMOS33 } [get_ports led3]   ;# LD3 green
set_property -dict { PACKAGE_PIN H5  IOSTANDARD LVCMOS33 } [get_ports led4]   ;# LD4
set_property -dict { PACKAGE_PIN J5  IOSTANDARD LVCMOS33 } [get_ports led5]   ;# LD5
set_property -dict { PACKAGE_PIN T9  IOSTANDARD LVCMOS33 } [get_ports led6]   ;# LD6
set_property -dict { PACKAGE_PIN T10 IOSTANDARD LVCMOS33 } [get_ports led7]   ;# LD7

set_property CFGBVS         VCCO    [current_design]
set_property CONFIG_VOLTAGE 3.3     [current_design]
set_property BITSTREAM.CONFIG.SPI_BUSWIDTH 4 [current_design]
