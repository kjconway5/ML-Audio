// pipeline_top.sv
//
// Chain: audio_in (16-bit, 16 kHz) --> STFFT --> LogMel --> bfpexp_comp --> spect_buffer
//
module pipeline_top #(
    // -----------------------------------------------------------------------
    // STFFT
    // -----------------------------------------------------------------------
    parameter int IW_STFFT  = 16,
    parameter int OW_STFFT  = 16,
    parameter int FFT_SIZE  = 256,
    parameter int HOP       = FFT_SIZE / 2,

    // -----------------------------------------------------------------------
    // LogMel
    // -----------------------------------------------------------------------
    parameter int IW_LOGMEL  = OW_STFFT,
    parameter int SHIFT      = 6,
    parameter int N_MELS     = 40,
    parameter int N_BINS     = 129,
    parameter int MAX_COEFFS = 16,
    parameter int POWER_W    = 2*IW_LOGMEL - SHIFT,  // 26
    parameter int WEIGHT_W   = 16,
    parameter int ACCUM_W    = 54,
    parameter int LOG_OUT_W  = 16,
    parameter int LUT_FRAC   = 6,
    parameter int OUT_W      = 16,

    // -----------------------------------------------------------------------
    // bfpexp compensation
    //
    // The R2FFT outputs FFT bins scaled by 2^(-bfpexp) (right-shifted to
    // prevent overflow).  This means logmel computes:
    //   log2(|scaled_X|^2) = log2(|true_X|^2) - 2*bfpexp
    // The correction adds back 2*bfpexp log2 units to each mel output.
    //
    // BFP_Q_FRAC must match the fractional-bit count of logmel's output word.
    // If the comparison script divides by (1 << Q_FRAC) to get float log2
    // values, set BFP_Q_FRAC = Q_FRAC (typically 10).
    // -----------------------------------------------------------------------
    parameter int BFP_Q_FRAC = 10,    // fractional bits in logmel OUT_W output

    // -----------------------------------------------------------------------
    // Spectrogram buffer
    // -----------------------------------------------------------------------
    parameter int SPECT_SHIFT = 4,
    parameter int N_FRAMES    = 50,
    parameter int ADDR_W      = 11
) (
    input  logic clk_i,
    input  logic reset_i,

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

    input  logic               flash_mel_coeff_we_i,
    input  logic [7:0]         flash_mel_coeff_addr_i,
    input  logic [15:0]        flash_mel_coeff_data_i,

    input  logic               flash_mel_index_we_i,
    input  logic [7:0]         flash_mel_index_addr_i,
    input  logic [7:0]         flash_mel_index_data_i,

    input  logic               flash_log_lut_we_i,
    input  logic [LUT_FRAC-1:0]  flash_log_lut_addr_i,
    input  logic [LOG_OUT_W-1:0] flash_log_lut_data_i
);

// ==========================================================================
// 1. STFFT
// ==========================================================================

logic [2*OW_STFFT-1:0] o_fft_result;
logic                   o_fft_sync;
logic                   win_ce_raw;
logic signed [7:0]      bfpexp_raw;   // per-frame BFP exponent from R2FFT

stfft #(
    .IW      (IW_STFFT),
    .OW      (OW_STFFT),
    .FFT_SIZE(FFT_SIZE),
    .HOP     (HOP)
) u_stfft (
    .i_clk       (clk_i),
    .i_reset     (reset_i),
    .i_ce        (valid_i),
    .i_sample    (data_i),
    .o_fft_result(o_fft_result),
    .o_fft_sync  (o_fft_sync),
    .win_ce_o    (win_ce_raw),
    .o_bfpexp    (bfpexp_raw)       // now used — not tied off
);

// ==========================================================================
// 2. FFT output pipeline registers + 2-cycle delayed sync
//
//    o_fft_sync fires 1 cycle before o_fft_result[0].
//    After 2 pipeline FFs, fft_sync_rr fires 1 cycle before fft_result_rr[0].
// ==========================================================================

logic                    fft_sync_r, fft_sync_rr;
logic [2*OW_STFFT-1:0]   fft_result_r, fft_result_rr;

always_ff @(posedge clk_i) begin
    if (reset_i) begin
        fft_sync_r    <= '0;
        fft_sync_rr   <= '0;
        fft_result_r  <= '0;
        fft_result_rr <= '0;
    end else begin
        fft_sync_r    <= o_fft_sync;
        fft_sync_rr   <= fft_sync_r;
        fft_result_r  <= o_fft_result;
        fft_result_rr <= fft_result_r;
    end
end

logic signed [OW_STFFT-1:0] fft_re, fft_im;
assign fft_re = fft_result_rr[2*OW_STFFT-1 : OW_STFFT];
assign fft_im = fft_result_rr[OW_STFFT-1   : 0];

// ==========================================================================
// 3. Bin counter — counts N_BINS bins per frame at full DMA clock rate
// ==========================================================================

localparam int CNT_W = $clog2(N_BINS + 1);
logic [CNT_W-1:0] bin_cnt_q;

always_ff @(posedge clk_i) begin
    if (reset_i)
        bin_cnt_q <= '0;
    else if (fft_sync_rr)
        bin_cnt_q <= CNT_W'(N_BINS);
    else if (bin_cnt_q > 0)
        bin_cnt_q <= bin_cnt_q - 1'b1;
end

logic fft_valid;
assign fft_valid = (bin_cnt_q > 0);

// ==========================================================================
// 4. bfpexp latch
//
//    Latch bfpexp at fft_sync_rr (the same cycle logmel receives its sync).
//    bfpexp_for_mel is stable for the entire logmel processing window because
//    it only updates when a new FFT frame begins.
// ==========================================================================

logic signed [7:0] bfpexp_for_mel;

always_ff @(posedge clk_i) begin
    if (reset_i)
        bfpexp_for_mel <= '0;
    else if (fft_sync_rr)        // update at the same time logmel sees sync
        bfpexp_for_mel <= bfpexp_raw;
end

// ==========================================================================
// 5. LogMel
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
    .fft_sync_il            (fft_sync_rr),
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
// 6. bfpexp compensation
//
//    The R2FFT right-shifts the FFT data by bfpexp bits to prevent overflow.
//    This means the stored FFT bins are 2^bfpexp times smaller than the true
//    values.  In log2 domain:
//
//      log2(|true_X|^2) = log2(|stored_X|^2) + 2 * bfpexp
//
//    logmel computes log2(|stored_X|^2), so we add 2*bfpexp log2 units.
//    With BFP_Q_FRAC fractional bits per log2 unit, the correction in
//    output-word units is:
//
//      correction = 2 * bfpexp * (1 << BFP_Q_FRAC)
//
//    Saturating unsigned add: clip to [0, 2^OUT_W - 1].
// ==========================================================================

// Correction value: signed arithmetic, wide enough to hold full range.
// bfpexp_for_mel is signed [7:0].  max |correction| = 2*127*(1<<10) = 260096
// Need 18 bits to represent this safely.
localparam int CORR_W = OUT_W + 10;  // enough headroom

logic signed [CORR_W-1:0] bfp_correction;
assign bfp_correction = ($signed(bfpexp_for_mel) <<< 1) * (1 << BFP_Q_FRAC);
// = 2 * bfpexp_for_mel * 2^BFP_Q_FRAC

logic signed [CORR_W-1:0] mel_compensated_wide;
assign mel_compensated_wide = $signed({{(CORR_W-OUT_W){1'b0}}, mel_data})
                            + bfp_correction;

logic [OUT_W-1:0] mel_compensated;
always_comb begin
    if (mel_compensated_wide < 0)
        mel_compensated = '0;                        // saturate low
    else if (mel_compensated_wide >= (1 << OUT_W))
        mel_compensated = {OUT_W{1'b1}};             // saturate high
    else
        mel_compensated = mel_compensated_wide[OUT_W-1:0];
end

// ==========================================================================
// 7. Spectrogram buffer  (receives bfpexp-compensated mel values)
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
    .cnn_data_i     (mel_compensated),   // compensated values
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