
module pipeline_top #(
    // stfft params
    parameter IW_STFFT = 14,
    parameter OW_STFFT = 18,
    parameter FFT_SIZE = 256,

    // logmel params
    parameter int IW_LOGMEL  = OW_STFFT,   // STFT output width
    parameter int SHIFT      = 6,          // power_calc shift
    parameter int N_MELS     = 40,         // mel bins
    parameter int N_BINS     = 129,        // FFT bins
    parameter int MAX_COEFFS = 16,         // sparse ROM depth
    parameter int POWER_W    = 31,         // power_calc output width = 2*IW - SHIFT + 1 - 1
    parameter int WEIGHT_W   = 16,         // mel coefficient width
    parameter int ACCUM_W    = 54,         // MAC accumulator width
    parameter int LOG_OUT_W  = 16,         // log_lut output width
    parameter int LUT_FRAC   = 6,
    parameter int OUT_W      = 16,         // output width to CNN

    //spect_buffer params 
    parameter int SPECT_SHIFT = 4,       // first_conv SPECT_SHIFT from export.py
    parameter int N_FRAMES    = 50,      // frames per inference window
    parameter int ADDR_W      = 11       // must match spectrogram_sram ADDR_W
) (
    input  logic clk_i,
    input  logic reset_i,


    // Audio input — 16 kHz, 16-bit signed PCM
    // (For the full pipeline with CIC+FIR, this would be the FIR output.)
    input  logic signed [15:0] data_i,
    input  logic               valid_i,

    // Block floating-point exponent from R2FFT (per FFT frame).
    // log-domain correction:  true_log_power = computed_log + 2*bfpexp*log2(2)
    output logic signed [7:0]  bfpexp_o,

    output logic                    spect_done,
    output logic                    spect_write_sel,

    // Flash write for mel coeff SRAM (sparse, 256 x 16-bit)
    input  logic [0:0]            flash_mel_coeff_we_i,
    input  logic [7:0]            flash_mel_coeff_addr_i,
    input  logic [15:0]           flash_mel_coeff_data_i,
 
    // Flash write for mel index SRAM (starts/ends/offsets, 256 x 8-bit)
    input  logic [0:0]            flash_mel_index_we_i,
    input  logic [7:0]            flash_mel_index_addr_i,
    input  logic [7:0]            flash_mel_index_data_i,
 
    // Flash write for log LUT SRAM (64 x 16-bit)
    input  logic [0:0]            flash_log_lut_we_i,
    input  logic [LUT_FRAC-1:0]   flash_log_lut_addr_i,
    input  logic [LOG_OUT_W-1:0]  flash_log_lut_data_i

    // Spectrogram SRAM — Bank B
    output wire                sp_b_we,
    output wire [ADDR_W-1:0]   sp_b_waddr,
    output wire signed [7:0]   sp_b_wdata,

    output logic               spect_done,
    output logic               spect_write_sel,

    // Flash-load ports for LogMel SRAMs (unchanged)
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

    
    localparam int CNT_W = $clog2(N_BINS + 1);
    logic [CNT_W-1:0] bin_cnt_q;

// Register bfpexp so it is stable and aligned with the pipeline output
always_ff @(posedge clk_i) begin
    if (reset_i) bfpexp_o <= '0;
    else if (o_fft_sync) bfpexp_o <= bfpexp_raw;  // latch at start of each frame
end

// 2.  FFT output pipeline registers
/*
     o_fft_sync fires 1 cycle before the first valid data word.
     We pipeline the result by 2 cycles (same as original) so:
       fft_sync_r  → delayed sync (used to load bin counter)
       fft_result_rr → data aligned with bin counter

     NOTE: win_ce_rr is kept only for debug/observation; it is NOT used
     to qualify fft_valid (see section 3).
*/

logic               fft_sync_r;
logic [2*OW_STFFT-1:0] fft_result_r, fft_result_rr;
logic               win_ce_r, win_ce_rr;  // debug only

always_ff @(posedge clk_i) begin
    if (reset_i) begin
        fft_sync_r    <= '0;
        fft_result_r  <= '0;
        fft_result_rr <= '0;
        win_ce_r      <= '0;
        win_ce_rr     <= '0;
    end else begin
        fft_sync_r    <= o_fft_sync;
        fft_result_r  <= o_fft_result;
        fft_result_rr <= fft_result_r;
        win_ce_r      <= win_ce_raw;
        win_ce_rr     <= win_ce_r;       // not used for gating — debug only
    end
end

// Split into 16-bit signed re/im rails for LogMel
logic signed [OW_STFFT-1:0] fft_re, fft_im;
assign fft_re = fft_result_rr[2*OW_STFFT-1 : OW_STFFT];  // [31:16]
assign fft_im = fft_result_rr[OW_STFFT-1   : 0];          // [15:0]


// 3.  Bin counter and fft_valid
/*
     CHANGE FROM ORIGINAL:
     The original used:   fft_valid = (bin_cnt_q > 0) && win_ce_rr

     The new stfft DMA readout streams all FFT_SIZE bins back-to-back at
     full clock speed with no win_ce gating on the output side.
     Therefore the counter must decrement every clock after sync, not
     every win_ce.

     We load N_BINS (129) rather than FFT_SIZE (256) because for a real
     input signal only bins 0..N/2 are unique.  Bins 129..255 are the
     conjugate mirror and are discarded.

     Timeline (relative to o_fft_sync):
       +0  sync fires, fft_sync_r fires next cycle
       +1  fft_sync_r fires → bin_cnt_q loads N_BINS (129)
       +2  bin_cnt_q = 129, fft_result_rr = bin[0]  → fft_valid = 1
       ...
       +130 bin_cnt_q = 1,   fft_result_rr = bin[128] → fft_valid = 1
       +131 bin_cnt_q = 0                              → fft_valid = 0
*/

localparam int CNT_W = $clog2(N_BINS + 1);
logic [CNT_W-1:0] bin_cnt_q;

always_ff @(posedge clk_i) begin
    if (reset_i)
        bin_cnt_q <= '0;
    else if (fft_sync_r)          // sync fires 1 cycle before data → load now
        bin_cnt_q <= CNT_W'(N_BINS);
    else if (bin_cnt_q > 0)       // count every clock — DMA streams at full rate
        bin_cnt_q <= bin_cnt_q - 1'b1;
end

logic fft_valid;
assign fft_valid = (bin_cnt_q > 0);  // no win_ce gating


// 4.  LogMel  (interface unchanged)

logic [OUT_W-1:0] mel_data;
logic             mel_valid;
logic             mel_ready;

logmel_top #(
    .IW        (IW_LOGMEL),   // 16
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
    .fft_sync_il            (fft_sync_r),
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


// 5.  Spectrogram buffer

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

    logic fft_valid;
    assign fft_valid = (bin_cnt_q > 0) && win_ce_rr;

    // Spectrogram output signals 
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
    ) logmel (
        .clk_i          (clk_i),
        .reset_i        (reset_i),

        // from STFT 
        .re_il          (fft_re),
        .im_il          (fft_im),
        .fft_valid_il   (fft_valid),
        .fft_sync_il    (fft_sync_r),  // 1-cycle delayed: arrives before the data

        // to CNN
        .cnn_data_ol    (mel_data),
        .cnn_valid_ol   (mel_valid),
        .cnn_ready_il   (mel_ready), // Driven by spect_buffer always high

        // Flash ports 
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


    spect_buffer_ctrl #(
        .SPECT_SHIFT(SPECT_SHIFT),
        .N_MELS     (N_MELS),
        .N_FRAMES   (N_FRAMES),
        .IN_W       (OUT_W),
        .ADDR_W     (ADDR_W)
    ) spectrogram_buffer (
        .clk          (clk_i),
        .reset        (reset_i),
        .cnn_data_i   (mel_data),
        .cnn_valid_i  (mel_valid),
        .cnn_ready_o  (mel_ready),
        .sp_a_we      (sp_a_we),
        .sp_a_waddr   (sp_a_waddr),
        .sp_a_wdata   (sp_a_wdata),
        .sp_b_we      (sp_b_we),
        .sp_b_waddr   (sp_b_waddr),
        .sp_b_wdata   (sp_b_wdata),
        .spect_done   (spect_done),
        .spect_write_sel(spect_write_sel)
    );


endmodule
