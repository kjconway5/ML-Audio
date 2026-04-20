// SPDX-FileCopyrightText: © 2025 XXX Authors
// SPDX-License-Identifier: Apache-2.0
 
`default_nettype none
 
module chip_core #(
    parameter NUM_INPUT_PADS,
    parameter NUM_BIDIR_PADS,
    parameter NUM_ANALOG_PADS
    )(
    `ifdef USE_POWER_PINS
    inout  wire VDD,
    inout  wire VSS,
    `endif
    
    input  wire clk,       // clock
    input  wire rst_n,     // reset (active low)
    
    input  wire [NUM_INPUT_PADS-1:0] input_in,   // Input value
    output wire [NUM_INPUT_PADS-1:0] input_pu,   // Pull-up
    output wire [NUM_INPUT_PADS-1:0] input_pd,   // Pull-down
 
    input  wire [NUM_BIDIR_PADS-1:0] bidir_in,   // Input value
    output wire [NUM_BIDIR_PADS-1:0] bidir_out,  // Output value
    output wire [NUM_BIDIR_PADS-1:0] bidir_oe,   // Output enable
    output wire [NUM_BIDIR_PADS-1:0] bidir_cs,   // Input type (0=CMOS Buffer, 1=Schmitt Trigger)
    output wire [NUM_BIDIR_PADS-1:0] bidir_sl,   // Slew rate (0=fast, 1=slow)
    output wire [NUM_BIDIR_PADS-1:0] bidir_ie,   // Input enable
    output wire [NUM_BIDIR_PADS-1:0] bidir_pu,   // Pull-up
    output wire [NUM_BIDIR_PADS-1:0] bidir_pd,   // Pull-down
 
    inout  wire [NUM_ANALOG_PADS-1:0] analog  // Analog
);
 
    // See here for usage: https://gf180mcu-pdk.readthedocs.io/en/latest/IPs/IO/gf180mcu_fd_io/digital.html
    
    // Disable pull-up and pull-down for input
    assign input_pu = '0;
    assign input_pd = '0;
 
    // Set the bidir as output
    assign bidir_oe = '1;
    assign bidir_cs = '0;
    assign bidir_sl = '0;
    assign bidir_ie = ~bidir_oe;
    assign bidir_pu = '0;
    assign bidir_pd = '0;
  
    logic _unused;                  // TODO: Set ununsed wires here 
    assign _unused = &bidir_in;
 
 
    wire reset; 
    assign reset = ~rst_n; 

    // SRAM loading signals 
    // active during bootup, will be driven by UART/SPI controller
 
    // Mel coeff SRAM flash (sparse coefficients, 256 x 16-bit)
    logic        flash_mel_coeff_we;
    logic [7:0]  flash_mel_coeff_addr;
    logic [15:0] flash_mel_coeff_data;
 
    // Mel index SRAM flash (starts/ends/offsets, 256 x 8-bit)
    logic        flash_mel_index_we;
    logic [7:0]  flash_mel_index_addr;
    logic [7:0]  flash_mel_index_data;
 
    // Log LUT SRAM flash (64 x 16-bit)
    logic        flash_log_lut_we;
    logic [5:0]  flash_log_lut_addr;
    logic [15:0] flash_log_lut_data;
 
    // TODO: Connect these to simple_flash instances 
    // ALSO: Must figure out how the 16-bit values will be driven from the 8-bit UART/SPI loading interface
    // For now, directly tie to zero (SRAMs will use $readmemh in simulation).
    assign flash_mel_coeff_we   = 1'b0;
    assign flash_mel_coeff_addr = 8'd0;
    assign flash_mel_coeff_data = 16'd0;
    assign flash_mel_index_we   = 1'b0;
    assign flash_mel_index_addr = 8'd0;
    assign flash_mel_index_data = 8'd0;
    assign flash_log_lut_we     = 1'b0;
    assign flash_log_lut_addr   = 6'd0;
    assign flash_log_lut_data   = 16'd0;

    // Spectrogram signals 
    // Bank A 
    wire            sp_a_we;
    wire [10:0]     sp_a_waddr;
    wire signed     [7:0] sp_a_wdata;

    // Bank B
    wire            sp_b_we;
    wire [10:0]     sp_b_waddr;
    wire signed     [7:0] sp_b_wdata;

    // Pipeline -> CNN handshake  
    wire            spect_done;
    wire            spect_write_sel;

    // KWS outputs 
    wire            kws_done; 
    wire [2:0]      kws_class_out; 

    pipeline_top #(
        .IW_STFFT   (14),
        .OW_STFFT   (18),
        .FFT_SIZE   (256),
        .N_MELS     (40),
        .N_BINS     (129),
        .N_FRAMES   (50),
        .SPECT_SHIFT(4), // update after retraining
        .ADDR_W     (11),
        .LUT_FRAC   (6)
    ) pipeline_inst (
        .clk_i              (clk),
        .reset_i            (reset),
        .data_i             ('0),           // TODO: connect to audio input pads
        .valid_i            (1'b0),         // TODO: connect to audio valid pad
        .sp_a_we            (sp_a_we),
        .sp_a_waddr         (sp_a_waddr),
        .sp_a_wdata         (sp_a_wdata),
        .sp_b_we            (sp_b_we),
        .sp_b_waddr         (sp_b_waddr),
        .sp_b_wdata         (sp_b_wdata),
        .spect_done         (spect_done),
        .spect_write_sel    (spect_write_sel),

        // Flash ports
        .flash_mel_coeff_we_i   (flash_mel_coeff_we),
        .flash_mel_coeff_addr_i (flash_mel_coeff_addr),
        .flash_mel_coeff_data_i (flash_mel_coeff_data),
        .flash_mel_index_we_i   (flash_mel_index_we),
        .flash_mel_index_addr_i (flash_mel_index_addr),
        .flash_mel_index_data_i (flash_mel_index_data),
        .flash_log_lut_we_i     (flash_log_lut_we),
        .flash_log_lut_addr_i   (flash_log_lut_addr),
        .flash_log_lut_data_i   (flash_log_lut_data)
    );

    kws_top kws_inst (
        .clk(clk),
        .reset(reset),
        .start(1'b0),               // TODO: connect to SERV or pad
        .done(kws_done),
        .class_out(kws_class_out),
        .cfg_we(1'b0),              // TODO: connect to SERV
        .cfg_addr('0),              // TODO: connect to SERV
        .cfg_wdata('0),             // TODO: connect to SERV
        .spect_done(spect_done),
        .spect_write_sel(spect_write_sel),
        .sp_a_we(sp_a_we),
        .sp_a_waddr(sp_a_waddr),
        .sp_a_wdata(sp_a_wdata),
        .sp_b_we(sp_b_we),
        .sp_b_waddr(sp_b_waddr),
        .sp_b_wdata(sp_b_wdata)
    );


    // TODO: Figure out output pad connection from kws_top -> bidir_out
    assign bidir_out = '0; 

    endmodule 
    `default_nettype wire 