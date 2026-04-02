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
 

    // Example SRAM instantiation from Wafer Space
    // logic [NUM_BIDIR_PADS-1:0] count;
    // always_ff @(posedge clk) begin
    //     if (!rst_n) begin
    //         count <= '0;
    //     end else begin
    //         if (&input_in) begin
    //             count <= count + 1;
    //         end
    //     end
    // end
 
    // logic [7:0] sram_0_out;
    // gf180mcu_fd_ip_sram__sram512x8m8wm1 sram_0 (
    //     `ifdef USE_POWER_PINS
    //     .VDD  (VDD),
    //     .VSS  (VSS),
    //     `endif
    //     .CLK  (clk),
    //     .CEN  (1'b1),
    //     .GWEN (1'b0),
    //     .WEN  (8'b0),
    //     .A    ('0),
    //     .D    ('0),
    //     .Q    (sram_0_out)
    // );
 
    // logic [7:0] sram_1_out;
    // gf180mcu_fd_ip_sram__sram512x8m8wm1 sram_1 (
    //     `ifdef USE_POWER_PINS
    //     .VDD  (VDD),
    //     .VSS  (VSS),
    //     `endif
    //     .CLK  (clk),
    //     .CEN  (1'b1),
    //     .GWEN (1'b0),
    //     .WEN  (8'b0),
    //     .A    ('0),
    //     .D    ('0),
    //     .Q    (sram_1_out)
    // );
 
    // assign bidir_out = count ^ {24'd0, sram_0_out, sram_1_out};
 
    wire reset; 
    assign reset = ~rst_n; 

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
        .IW_STFFT(14),
        .OW_STFFT(18),
        .FFT_SIZE(256),
        .N_MELS(40),
        .N_BINS(129),
        .N_FRAMES(50),
        .SPECT_SHIFT(4),    // update this after retraining
        .ADDR_W(11)
    ) 
    pipeline_inst (
        .clk_i(clk),
        .reset_i(reset),
        .data_i('0),          // TODO: connect to audio input pads
        .valid_i(1'b0),        // TODO: connect to audio valid pad
        .sp_a_we(sp_a_we),
        .sp_a_waddr(sp_a_waddr),
        .sp_a_wdata(sp_a_wdata),
        .sp_b_we(sp_b_we),
        .sp_b_waddr(sp_b_waddr),
        .sp_b_wdata(sp_b_wdata),
        .spect_done(spect_done),
        .spect_write_sel(spect_write_sel)
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