module logmel_top #(
    parameter int IW         = 18,   // STFT output width
    parameter int SHIFT      = 6,    // power_calc shift
    parameter int N_MELS     = 40,   // mel bins
    parameter int N_BINS     = 129,  // FFT bins
    parameter int MAX_COEFFS = 16,   // sparse ROM depth
    parameter int POWER_W    = 31,   // power_calc output width = 2*IW - SHIFT + 1 - 1
    parameter int WEIGHT_W   = 16,   // mel coefficient width
    parameter int ACCUM_W    = 54,   // MAC accumulator width
    parameter int LOG_OUT_W  = 16,   // log_lut output width
    parameter int LUT_FRAC   = 6,    // log2 LUT fractional bits
    parameter int OUT_W      = 16    // output width to CNN
)(
    input  logic [0:0] clk_i,
    input  logic [0:0] reset_i,

    // from STFT
    input  logic [IW-1:0]       re_il,        // real FFT output
    input  logic [IW-1:0]       im_il,        // imaginary FFT output
    input  logic [0:0]          fft_valid_il, // per-sample valid
    input  logic [0:0]          fft_sync_il,  // frame sync pulse 

    // to CNN
    output logic [OUT_W-1:0]    cnn_data_ol,  // log-mel feature value
    output logic [0:0]          cnn_valid_ol, // valid handshake to CNN
    input  logic [0:0]          cnn_ready_il,  // backpressure from CNN

    // Flash write for mel coeff SRAM (sparse, 256 x 16-bit via two sram256x8)
    input  logic [0:0]          flash_mel_coeff_we_i,
    input  logic [7:0]          flash_mel_coeff_addr_i,
    input  logic [15:0]         flash_mel_coeff_data_i,
 
    // Flash write for mel index SRAM (starts/ends/offsets, 256 x 8-bit)
    input  logic [0:0]          flash_mel_index_we_i,
    input  logic [7:0]          flash_mel_index_addr_i,
    input  logic [7:0]          flash_mel_index_data_i,
 
    // Flash write for log LUT SRAM (64 x 16-bit via two sram64x8)
    input  logic [0:0]          flash_log_lut_we_i,
    input  logic [LUT_FRAC-1:0] flash_log_lut_addr_i,
    input  logic [LOG_OUT_W-1:0] flash_log_lut_data_i,

    // Test mode for SRAM readback
    input  logic                test_mode_i,

    input  logic [7:0]          test_coeff_addr_i,
    input  logic [7:0]          test_index_addr_i,
    input  logic [LUT_FRAC-1:0] test_lut_addr_i,

    // New port for VAD
    input logic [31:0]          vad_threshold_il,

    // VAD output to spect_buffer
    output logic                vad_active_ol,

    // DFT / debug
    input  logic                dft_vad_obs_en_i,
    output logic                vad_frame_drop_ol,
    output logic [$clog2(N_MELS)-1:0] mel_idx_test,
    output logic [(2*IW)-SHIFT:0] power_test,
    output logic [1:0] frame_control_state
);

    // Internal Signals
    // connect power_calc to mel_filterbank
    logic [POWER_W-1:0]   power;
    logic [0:0]    power_valid;
    // connect mel_filterbank to frame_controller + log_lut
    logic [N_MELS-1:0][ACCUM_W-1:0] mel_energy;
    logic [0:0]   filterbank_done;      
    // connect frame_controller to log_lut + output_buffer
    logic [$clog2(N_MELS)-1:0] mel_idx;
    logic [0:0] log_en;
    logic [0:0] output_valid;
    // connect log_lut to output_buffer
    logic [N_MELS-1:0][LOG_OUT_W-1:0] log_out;  // all 40 log values
    logic [0:0] log_done;          // all 40 compressed
    // connect output_buffer to frame_controller
    logic [0:0] frame_sent;

    // power_calc
    // takes re/im from STFT, outputs power per bin
    power_calc #(
        .IW(IW),
        .SHIFT(SHIFT)
    ) u_power_calc (
        .clk      (clk_i),
        .reset_i  (reset_i),
        .real_il  (re_il),
        .imag_il  (im_il),
        .valid_il (fft_valid_il),
        .power_ol (power),
        .valid_ol (power_valid),
        .power_test (power_test),
        .test_mode_audio (test_mode_i)
    );

    // spectral_vad signals and instantiation
    logic        vad_active;
    logic        vad_done;

spectral_vad #(
    .POWER_W  (POWER_W),
    .N_BINS   (N_BINS),
    .THRESH_W (32)
    ) u_vad (
    .clk_i              (clk_i),
    .reset_i            (reset_i),
    .power_il           (power),
    .power_valid_il     (power_valid),
    .fft_sync_il        (fft_sync_il),
    .threshold_il       (vad_threshold_il),
    .dft_vad_obs_en_i   (dft_vad_obs_en_i), 
    .vad_frame_drop_ol  (vad_frame_drop_ol),
    .voice_active_ol    (vad_active),
    .vad_done_ol        (vad_done)
    );

    // mel_filterbank
    // accumulates power*weight for all 40 filters across 129 bins
    // self-manages bin counter, pulses valid_ol when frame complete
    mel_filterbank #(
        .N_MELS     (N_MELS),
        .N_BINS     (N_BINS),
        .MAX_COEFFS (MAX_COEFFS),
        .POWER_W    (POWER_W),
        .WEIGHT_W   (WEIGHT_W),
        .ACCUM_W    (ACCUM_W)
    ) u_mel_filterbank (
        .clk_i               (clk_i),
        .reset_i             (reset_i),
        .power_il            (power),
        .valid_il            (power_valid),
        .mel_ol              (mel_energy),
        .valid_ol            (filterbank_done),
        .flash_coeff_we_i    (flash_mel_coeff_we_i),
        .flash_coeff_addr_i  (flash_mel_coeff_addr_i),
        .flash_coeff_data_i  (flash_mel_coeff_data_i),
        .flash_index_we_i    (flash_mel_index_we_i),
        .flash_index_addr_i  (flash_mel_index_addr_i),
        .flash_index_data_i  (flash_mel_index_data_i),
        .test_mode_i         (test_mode_i),
        .test_coeff_addr_i   (test_coeff_addr_i),
        .test_index_addr_i   (test_index_addr_i)
    );


    // frame_controller
    // FSM to sequence log compression and output stages
    // filterbank manages its own accumulation so FSM just waits for done
    frame_control #(
        .MEL_BINS(N_MELS)
    ) u_frame_ctrl (
        .clk               (clk_i),
        .reset             (reset_i),
        .fft_sync_i        (fft_sync_il),
        .filterbank_done_i (filterbank_done),
        .vad_active_i      (vad_active),
        .frame_sent_i      (frame_sent),
        .mel_idx_o         (mel_idx),
        .log_en_o          (log_en),
        .output_valid_o    (output_valid),
        .test_mode_audio   (test_mode_i),
        .mel_idx_test      (mel_idx_test),
        .frame_control_state (frame_control_state)
    );

    // log_lut
    // compresses 40 mel energies one per cycle using Log2 IP + LUT
    // mel_idx steps 0->39 driven by frame_controller during LOG_COMPRESS state
    log_lut #(
        .ACCUM_W   (ACCUM_W),
        .N_MELS    (N_MELS),
        .LOG_OUT_W (LOG_OUT_W),
        .LUT_FRAC  (LUT_FRAC),
        .Q_FRAC    (10) // changed from 12 to 10 for 6.10 because it was saturating very easily with 12 fractional bits
    ) u_log_lut (
        .clk_i                 (clk_i),
        .reset_i               (reset_i),
        .mel_energy_i        (mel_energy), // from mel filterbank
        .mel_idx_i           (mel_idx), // from frame controller
        .log_en_i            (log_en), // from frame controller
        .log_out_o           (log_out), // to output buffer
        .log_done_o          (log_done), // to frame controller
        // Flash log LUT SRAM
        .flash_write_enable_i (flash_log_lut_we_i),
        .flash_addr_i         (flash_log_lut_addr_i),
        .flash_write_data_i   (flash_log_lut_data_i),
        .test_mode_i          (test_mode_i),
        .test_addr_i          (test_lut_addr_i)
    );

    // output_buffer
    // holds 40 log values, handshakes with CNN 
    // fires frame_sent back to FSM when all 40 sent
    output_buffer #(
        .N_MELS(N_MELS),
        .OUT_W (OUT_W)
    ) u_output_buffer (
        .clk          (clk_i),
        .reset        (reset_i),
        .log_out_i    (log_out),
        .load_i       (log_done),     
        .cnn_data_o   (cnn_data_ol),
        .cnn_valid_o  (cnn_valid_ol),
        .cnn_ready_i  (cnn_ready_il),
        .frame_sent_o (frame_sent)
    );


    assign vad_active_ol = vad_active;

endmodule
