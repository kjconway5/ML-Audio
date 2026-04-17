read_verilog ../top.sv
read_verilog ../../FIR/fastfir.v
read_verilog ../../FIR/firtap.v
read_verilog ../../CIC/cic_decimator.v

synth_design -top top -part xc7a35tcsg324-1

opt_design
place_design
route_design

write_bitstream -force top.bit