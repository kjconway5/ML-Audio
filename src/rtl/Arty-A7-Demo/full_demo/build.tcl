# build.tcl — non-project Vivado flow for full_demo_top on Arty A7-100t.
#
#   vivado -mode batch -source build.tcl
#
# Produces build/full_demo_top.bit.

set part xc7a100tcsg324-1
set top  full_demo_top

set demo_dir [file normalize [file dirname [info script]]]
set rtl_root [file normalize $demo_dir/../..]
set flash    $rtl_root/flash
set top_dir  $rtl_root/top
set logmel   $rtl_root/Log-Mel
set stfft    $rtl_root/STFFT
set spect    $rtl_root/SPECT_BUFFER
set dscnn    $rtl_root/dscnn
set stage2   $rtl_root/Arty-A7-Demo/features_stage2

set build_dir $demo_dir/build
file mkdir $build_dir

# Hex assets — $readmemh resolves against the working directory.
# `hanning.hex` is load-bearing (Hann window ROM hardcoded into stfft.sv);
# log2_lut / mel_coeffs / mel_indices initialize behavioral SRAMs in SIM
# mode and are overwritten at boot time anyway.
foreach src [list \
    $stfft/ZipCPU/hanning.hex \
    $logmel/data/log2_lut.hex \
    $logmel/data/mel_coeffs_sparse.hex \
    $logmel/data/mel_indices.hex \
] {
    file copy -force $src $demo_dir/[file tail $src]
}

# Sources

set sv_sources [list \
    $flash/boot_pkg.sv \
    $flash/boot_controller.sv \
    $flash/features_boot_router.sv \
    $flash/dscnn_boot_router.sv \
    $top_dir/pipeline_top.sv \
    $stfft/rtl/stfft.sv \
    $stfft/rtl/fft_data_ram.sv \
    $stfft/rtl/fft_twiddle_rom.sv \
    $logmel/rtl/log_top/logmel_top.sv \
    $logmel/rtl/power_calc/power_calc.sv \
    $logmel/rtl/mel_filterbank/mel_filterbank.sv \
    $logmel/rtl/mel_filterbank/mel_coeff_sram.sv \
    $logmel/rtl/mac_unit/mac_unit.sv \
    $logmel/rtl/frame_control/frame_control.sv \
    $logmel/rtl/log_lut/log_lut.sv \
    $logmel/rtl/log_lut/log_lut_sram.sv \
    $logmel/rtl/output_buffer/output_buffer.sv \
    $spect/spect_buffer_ctrl.sv \
    $dscnn/kws_top.sv \
    $dscnn/fsm/FSM.sv \
    $dscnn/mac_array/mac_array.sv \
    $dscnn/requant/requant.sv \
    $dscnn/bias_dff/bias_DFFs.sv \
    $dscnn/feature_sram/feature_sram.sv \
    $dscnn/weight_sram/weight_sram.sv \
    $dscnn/spectrogram_sram/spectrogram_sram.sv \
    $stage2/spect_capture.sv \
    $stage2/spect_streamer.sv \
    $demo_dir/class_reporter.sv \
    $demo_dir/full_demo_top.sv \
]

foreach f [glob -nocomplain $logmel/ip/*.sv]      { lappend sv_sources $f }
foreach f [glob -nocomplain $stfft/R2FFT/hdl/*.sv] { lappend sv_sources $f }

set v_sources [list \
    $flash/uart.v \
    $flash/uart_rx.v \
    $flash/uart_tx.v \
]

set xdc_sources [list $demo_dir/arty_full.xdc]

# Flow

create_project -in_memory -part $part

read_verilog -sv $sv_sources
read_verilog       $v_sources
read_xdc          $xdc_sources

set_property top $top [current_fileset]

synth_design -top $top -part $part -verilog_define SIM
write_checkpoint -force $build_dir/post_synth.dcp

opt_design
place_design
route_design
write_checkpoint -force $build_dir/post_route.dcp

report_timing_summary -file $build_dir/timing_summary.rpt
report_utilization     -file $build_dir/utilization.rpt
report_drc             -file $build_dir/drc.rpt

write_bitstream -force $build_dir/$top.bit
puts "INFO: wrote $build_dir/$top.bit"
