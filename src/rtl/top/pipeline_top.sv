// pipeline_top.sv
//
// Chain: audio_in (16-bit, 16 kHz) --> STFFT --> LogMel --> spect_buffer
//
module pipeline_top #(
    // -----------------------------------------------------------------------
    // STFFT
    // -----------------------------------------------------------------------
    parameter int IW_STFFT  = 16,
    parameter int OW_STFFT  = 16,
    parameter int FFT_SIZE  = 256,

    // -----------------------------------------------------------------------
    // LogMel
    // -----------------------------------------------------------------------
    parameter int IW_LOGMEL  = OW_STFFT,           // 16
    parameter int SHIFT      = 6,
    parameter int N_MELS     = 40,
    parameter int N_BINS     = 129,                 // FFT/2 + 1 (real signal)
    parameter int MAX_COEFFS = 16,
    parameter int POWER_W    = 2*IW_LOGMEL - SHIFT, // 26
    parameter int WEIGHT_W   = 16,
    parameter int ACCUM_W    = 54,
    parameter int LOG_OUT_W  = 16,
    parameter int LUT_FRAC   = 6,
    parameter int OUT_W      = 16,

    // -----------------------------------------------------------------------
    // Spectrogram buffer
    // -----------------------------------------------------------------------
    parameter int SPECT_SHIFT = 4,
    parameter int N_FRAMES    = 50,
    parameter int ADDR_W      = 11
) (
    input  logic clk_i,
    input  logic reset_i,

    // Audio input — 16 kHz, 16-bit signed PCM
    input  logic signed [15:0] data_i,
    input  logic               valid_i,

    // Spectrogram SRAM — Bank A
    output wire                sp_a_we,
    output wire [ADDR_W-1:0]   sp_a_waddr,
    output wire signed [7:0]   sp_a_wdata,

    // Spectrogram SRAM — Bank B
    output wire                sp_b_we,
    output wire [ADDR_W-1:0]   sp_b_waddr,
    output wire signed [7:0]   sp_b_wdata,

    output logic               spect_done,
    output logic               spect_write_sel,

    // Flash-load: mel coefficient SRAM
    input  logic               flash_mel_coeff_we_i,
    input  logic [7:0]         flash_mel_coeff_addr_i,
    input  logic [15:0]        flash_mel_coeff_data_i,

    // Flash-load: mel index SRAM
    input  logic               flash_mel_index_we_i,
    input  logic [7:0]         flash_mel_index_addr_i,
    input  logic [7:0]         flash_mel_index_data_i,

    // Flash-load: log LUT SRAM
    input  logic               flash_log_lut_we_i,
    input  logic [LUT_FRAC-1:0]  flash_log_lut_addr_i,
    input  logic [LOG_OUT_W-1:0] flash_log_lut_data_i
);

// ==========================================================================
// 1. STFFT  (R2FFT-based, Hanning window, 256-point)
//
//    Timing contract from stfft.sv:
//      - o_fft_sync  : 1-cycle pulse, fires 1 cycle BEFORE the first valid
//                      o_fft_result word (DMA readout then streams all
//                      FFT_SIZE bins back-to-back at full clock speed).
//      - win_ce_o    : windowed INPUT sample strobe — NOT an output valid.
//      - o_bfpexp    : block-FP exponent (unused here, tied to wire).
// ==========================================================================

logic [2*OW_STFFT-1:0] o_fft_result;   // {re[15:0], im[15:0]}
logic                   o_fft_sync;
logic                   win_ce_raw;
/* verilator lint_off UNUSED */
logic signed [7:0]      bfpexp_nc;
/* verilator lint_on UNUSED */

stfft #(
    .IW      (IW_STFFT),
    .OW      (OW_STFFT),
    .FFT_SIZE(FFT_SIZE)
) u_stfft (
    .i_clk       (clk_i),
    .i_reset     (reset_i),
    .i_ce        (valid_i),
    .i_sample    (data_i),
    .o_fft_result(o_fft_result),
    .o_fft_sync  (o_fft_sync),
    .win_ce_o    (win_ce_raw),
    .o_bfpexp    (bfpexp_nc)
);

// ==========================================================================
// 2. FFT output pipeline registers
//
//    o_fft_sync fires 1 cycle before o_fft_result[0].
//    Data passes through 2 pipeline FFs → fft_result_rr is 2 cycles behind
//    o_fft_result.  Therefore the sync must ALSO be delayed 2 cycles
//    (fft_sync_rr) so that it still arrives exactly 1 cycle before the
//    corresponding data appears in fft_result_rr.
//
//    Timeline (T=0 = cycle o_fft_sync pulses):
//      T=0  o_fft_sync=1,   o_fft_result=stale
//      T=1  o_fft_sync=0,   o_fft_result=bin[0],  fft_sync_r=1
//      T=2  o_fft_result=bin[1], fft_result_r=bin[0], fft_sync_r=0,  fft_sync_rr=1
//      T=3  fft_result_r=bin[1], fft_result_rr=bin[0], fft_sync_rr=0
//                                ↑ first valid data in fft_result_rr
//                           ↑ fft_sync_rr fired 1 cycle earlier — correct for logmel
// ==========================================================================

logic                    fft_sync_r, fft_sync_rr;   // 1- and 2-cycle delayed sync
logic [2*OW_STFFT-1:0]   fft_result_r, fft_result_rr;

always_ff @(posedge clk_i) begin
    if (reset_i) begin
        fft_sync_r    <= '0;
        fft_sync_rr   <= '0;
        fft_result_r  <= '0;
        fft_result_rr <= '0;
    end else begin
        fft_sync_r    <= o_fft_sync;     // 1-cycle delay
        fft_sync_rr   <= fft_sync_r;    // 2-cycle delay  ← key fix
        fft_result_r  <= o_fft_result;
        fft_result_rr <= fft_result_r;
    end
end

// Split 32-bit bus into two 16-bit signed rails
logic signed [OW_STFFT-1:0] fft_re, fft_im;
assign fft_re = fft_result_rr[2*OW_STFFT-1 : OW_STFFT];  // [31:16]
assign fft_im = fft_result_rr[OW_STFFT-1   : 0];          // [15:0]

// ==========================================================================
// 3. Bin counter and fft_valid
//
//    Load on fft_sync_rr (T=2) with N_BINS so that:
//      - T=3: bin_cnt=N_BINS, fft_result_rr=bin[0]  → fft_valid=1  ✓
//      - T=N_BINS+2: bin_cnt=1, fft_result_rr=bin[N_BINS-1] → fft_valid=1 ✓
//      - T=N_BINS+3: bin_cnt=0 → fft_valid=0 ✓
//    Exactly N_BINS valid cycles, all containing real bin data.
//
//    We count only N_BINS=129 (the unique bins for a real signal);
//    bins 129-255 are the conjugate mirror and are discarded.
// ==========================================================================

localparam int CNT_W = $clog2(N_BINS + 1);
logic [CNT_W-1:0] bin_cnt_q;

always_ff @(posedge clk_i) begin
    if (reset_i)
        bin_cnt_q <= '0;
    else if (fft_sync_rr)           // ← 2-cycle delayed sync, load N_BINS (not N_BINS-1)
        bin_cnt_q <= CNT_W'(N_BINS);
    else if (bin_cnt_q > 0)
        bin_cnt_q <= bin_cnt_q - 1'b1;
end

logic fft_valid;
assign fft_valid = (bin_cnt_q > 0);

// ==========================================================================
// 4. LogMel
//    fft_sync_il = fft_sync_rr  (fires 1 cycle before first valid fft_re/im)
// ==========================================================================

logic [OUT_W-1:0] mel_data;
logic             mel_valid;
logic             mel_ready;

logmel_top #(
    .IW        (IW_LOGMEL),
    .SHIFT     (SHIFT),
    .N_MELS    (N_MELS),
    .N_BINS    (N_BINS),
    .MAX_COEFFS(MAX_COEFFS),
    .POWER_W   (POWER_W),
    .WEIGHT_W  (WEIGHT_W),
    .ACCUM_W   (ACCUM_W),
    .LOG_OUT_W (LOG_OUT_W),
    .LUT_FRAC  (LUT_FRAC),
    .OUT_W     (OUT_W)
) u_logmel (
    .clk_i                  (clk_i),
    .reset_i                (reset_i),
    .re_il                  (fft_re),
    .im_il                  (fft_im),
    .fft_valid_il           (fft_valid),
    .fft_sync_il            (fft_sync_rr),   // ← 2-cycle delayed, 1 cycle before data
    .cnn_data_ol            (mel_data),
    .cnn_valid_ol           (mel_valid),
    .cnn_ready_il           (mel_ready),
    .flash_mel_coeff_we_i   (flash_mel_coeff_we_i),
    .flash_mel_coeff_addr_i (flash_mel_coeff_addr_i),
    .flash_mel_coeff_data_i (flash_mel_coeff_data_i),
    .flash_mel_index_we_i   (flash_mel_index_we_i),
    .flash_mel_index_addr_i (flash_mel_index_addr_i),
    .flash_mel_index_data_i (flash_mel_index_data_i),
    .flash_log_lut_we_i     (flash_log_lut_we_i),
    .flash_log_lut_addr_i   (flash_log_lut_addr_i),
    .flash_log_lut_data_i   (flash_log_lut_data_i)
);

// ==========================================================================
// 5. Spectrogram buffer
// ==========================================================================

spect_buffer_ctrl #(
    .SPECT_SHIFT(SPECT_SHIFT),
    .N_MELS     (N_MELS),
    .N_FRAMES   (N_FRAMES),
    .IN_W       (OUT_W),
    .ADDR_W     (ADDR_W)
) u_spect_buf (
    .clk            (clk_i),
    .reset          (reset_i),
    .cnn_data_i     (mel_data),
    .cnn_valid_i    (mel_valid),
    .cnn_ready_o    (mel_ready),
    .sp_a_we        (sp_a_we),
    .sp_a_waddr     (sp_a_waddr),
    .sp_a_wdata     (sp_a_wdata),
    .sp_b_we        (sp_b_we),
    .sp_b_waddr     (sp_b_waddr),
    .sp_b_wdata     (sp_b_wdata),
    .spect_done     (spect_done),
    .spect_write_sel(spect_write_sel)
);

endmodule