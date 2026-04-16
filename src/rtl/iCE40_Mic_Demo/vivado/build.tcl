read_verilog ../rtl/top.sv
read_verilog ../rtl/fastfir.v
read_verilog ../rtl/firtap.v
read_verilog ../rtl/cic_decimator.v

synth_design -top top -part xc7a35tcsg324-1

opt_design
place_design
route_design

write_bitstream -force top.bit