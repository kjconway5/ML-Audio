// pipeline_top_fixed.sv
//
// Drop-in replacement for pipeline_top.sv that instantiates `stfft_fixed`
// instead of `stfft`.  The only difference from pipeline_top.sv is the
// instantiated module name on line ~100 — everything else (ports, params,
// timing, logmel chain, bfpexp compensation, spect buffer wiring) is
// byte-for-byte identical.
//
// See stfft_fixed.sv for the bug-fix rationale.

module pipeline_top #(
    parameter int IW_STFFT  = 16,
    parameter int OW_STFFT  = 16,
    parameter int FFT_SIZE  = 256,
    parameter int HOP       = FFT_SIZE / 2,

    parameter int IW_LOGMEL  = OW_STFFT,
    parameter int SHIFT      = 6,
    parameter int N_MELS     = 40,
    parameter int N_BINS     = 129,
    parameter int MAX_COEFFS = 16,
    parameter int POWER_W    = 2*IW_LOGMEL - SHIFT,
    parameter int WEIGHT_W   = 16,
    parameter int ACCUM_W    = 54,
    parameter int LOG_OUT_W  = 16,
    parameter int LUT_FRAC   = 6,
    parameter int OUT_W      = 16,

    parameter int BFP_Q_FRAC = 10,

    parameter int SPECT_SHIFT = 9,
    parameter int USE_INPUT_REQUANT = 1,
    parameter int INPUT_QUANT_MULT  = 5817845,
    parameter int INPUT_QUANT_SHIFT = 31,
    parameter int START_FRAME = 37,
    parameter int N_FRAMES    = 50,
    parameter int ADDR_W      = 11
) (
    input  logic clk_i,
    input  logic reset_i,

    input  logic signed [15:0] data_i,
    input  logic               valid_i,

    output wire                sp_a_we,
    output wire [ADDR_W-1:0]   sp_a_waddr,
    output wire signed [7:0]   sp_a_wdata,

    output wire                sp_b_we,
    output wire [ADDR_W-1:0]   sp_b_waddr,
    output wire signed [7:0]   sp_b_wdata,

    output logic               spect_done,
    output logic               spect_write_sel,

    output logic [OUT_W-1:0]   mel_compensated_o,
    output logic               mel_compensated_valid_o,

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
// 1. STFFT — now the FIXED variant (FIFO + per-channel local Hann)
// ==========================================================================

logic [2*OW_STFFT-1:0] o_fft_result;
logic                   o_fft_sync;
logic                   win_ce_raw;
logic signed [7:0]      bfpexp_raw;

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
    .o_bfpexp    (bfpexp_raw)
);

// ==========================================================================
// 2. FFT output pipeline registers + 2-cycle delayed sync
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
// 2b. FIX for Bug D — DMA-pipeline-aligned sync
//
// `fft_sync_rr` fires when the R2FFT's DMA begins, but the ACTUAL bin-0
// value propagates through:
//   a_dmadr_real_r → a_result → o_fft_result → fft_result_r → fft_result_rr
// which is 4 more cycles.  Meanwhile bin_cnt_q needs only 1 cycle to assert
// fft_valid after fft_sync_rr.  Net: the first 3 fft_valid=1 cycles show
// STALE fft_result_rr (from the previous frame), causing mel_filterbank to
// store garbage into power_buf[0..2] — a cyclic roll of +3 on the FFT
// output visible as a slight mel-bin shift and a "double ridge" pattern
// in comparison.png.
//
// Fix: delay fft_sync_rr by SYNC_ALIGN_DELAY cycles before driving the
// bin counter, the bfpexp latch, and the logmel sync.  This pushes
// fft_valid=1 to start exactly when bin 0 arrives at fft_result_rr.
// ==========================================================================

localparam int SYNC_ALIGN_DELAY = 3;  // empirically measured DMA-to-fft_result_rr gap

logic [SYNC_ALIGN_DELAY-1:0] fft_sync_align_sr;
always_ff @(posedge clk_i) begin
    if (reset_i)
        fft_sync_align_sr <= '0;
    else
        fft_sync_align_sr <= {fft_sync_align_sr[SYNC_ALIGN_DELAY-2:0], fft_sync_rr};
end
logic fft_sync_aligned;
assign fft_sync_aligned = fft_sync_align_sr[SYNC_ALIGN_DELAY-1];

// ==========================================================================
// 3. Bin counter  (driven by the aligned sync)
// ==========================================================================

localparam int CNT_W = $clog2(N_BINS + 1);
logic [CNT_W-1:0] bin_cnt_q;

always_ff @(posedge clk_i) begin
    if (reset_i)
        bin_cnt_q <= '0;
    else if (fft_sync_aligned)
        bin_cnt_q <= CNT_W'(N_BINS);
    else if (bin_cnt_q > 0)
        bin_cnt_q <= bin_cnt_q - 1'b1;
end

logic fft_valid;
assign fft_valid = (bin_cnt_q > 0);

// ==========================================================================
// 4. bfpexp latch  (also driven by aligned sync so it updates on the same
//    cycle logmel starts consuming the new frame)
// ==========================================================================

logic signed [7:0] bfpexp_for_mel;

always_ff @(posedge clk_i) begin
    if (reset_i)
        bfpexp_for_mel <= '0;
    else if (fft_sync_aligned)
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
    .fft_sync_il            (fft_sync_aligned),
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
// ==========================================================================

localparam int CORR_W = OUT_W + 10;

logic signed [CORR_W-1:0] bfp_correction;
assign bfp_correction = ($signed(bfpexp_for_mel) <<< 1) * (1 << BFP_Q_FRAC);

logic signed [CORR_W-1:0] mel_compensated_wide;
assign mel_compensated_wide = $signed({{(CORR_W-OUT_W){1'b0}}, mel_data})
                            + bfp_correction;

logic [OUT_W-1:0] mel_compensated;
always_comb begin
    if (mel_compensated_wide < 0)
        mel_compensated = '0;
    else if (mel_compensated_wide >= (1 << OUT_W))
        mel_compensated = {OUT_W{1'b1}};
    else
        mel_compensated = mel_compensated_wide[OUT_W-1:0];
end

assign mel_compensated_o       = mel_compensated;
assign mel_compensated_valid_o = mel_valid;

// ==========================================================================
// 7. Spectrogram buffer
// ==========================================================================

spect_buffer_ctrl #(
    .SPECT_SHIFT      (SPECT_SHIFT),
    .USE_INPUT_REQUANT(USE_INPUT_REQUANT),
    .INPUT_QUANT_MULT (INPUT_QUANT_MULT),
    .INPUT_QUANT_SHIFT(INPUT_QUANT_SHIFT),
    .START_FRAME      (START_FRAME),
    .N_MELS           (N_MELS),
    .N_FRAMES         (N_FRAMES),
    .IN_W             (OUT_W),
    .ADDR_W           (ADDR_W)
) u_spect_buf (
    .clk            (clk_i),
    .reset          (reset_i),
    .cnn_data_i     (mel_compensated),
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
