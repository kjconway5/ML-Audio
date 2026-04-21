
module pipeline_top #(
    // STFFT
    parameter int IW_STFFT  = 16,
    parameter int OW_STFFT  = 16,
    parameter int FFT_SIZE  = 256,

    // LogMel
    parameter int IW_LOGMEL  = OW_STFFT,          // 16
    parameter int SHIFT      = 6,
    parameter int N_MELS     = 40,
    parameter int N_BINS     = 129,                // FFT/2 + 1 (real signal)
    parameter int MAX_COEFFS = 16,
    parameter int POWER_W    = 2*IW_LOGMEL - SHIFT, // 26
    parameter int WEIGHT_W   = 16,
    parameter int ACCUM_W    = 54,
    parameter int LOG_OUT_W  = 16,
    parameter int LUT_FRAC   = 6,
    parameter int OUT_W      = 16,

    // Spectrogram buffer
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


// 1. STFFT  (R2FFT Core, Hanning window, 256-point)
/*
    Timing contract:
      - o_fft_sync  : 1-cycle pulse when DMA readout begins,
                      1 cycle BEFORE the first valid o_fft_result word.
      - o_fft_result: valid for FFT_SIZE consecutive clocks at full speed
                      after o_fft_sync (no gaps, no win_ce gating).
      - win_ce_o    : windowed INPUT sample strobe — NOT an output valid.
      - o_bfpexp    : block-FP exponent for the current frame (stable
                      during readout).  Ignored

    bfpexp: the R2FFT shifts internally to prevent overflow and
    reports the shift amount in bfpexp.  For log-mel the correction is
    purely additive in the log domain (+2*bfpexp*log2).  We tie it to an
    unused wire for now; compensate in post-processing or CNN pre-proc.
*/

logic [2*OW_STFFT-1:0] o_fft_result;  // {re[15:0], im[15:0]}
logic                   o_fft_sync;
logic                   win_ce_raw;    // windowed input CE — debug only
/* verilator lint_off UNUSED */
logic signed [7:0]      bfpexp_unused;
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
    .o_bfpexp    (bfpexp_unused)
);


// 2. FFT output pipeline registers
/*
    o_fft_sync fires 1 cycle before the first data word.
    Two-cycle pipeline delay keeps the same alignment as the original.
      fft_sync_r   — used to load the bin counter
      fft_result_rr — data aligned with active bin_cnt_q
*/

logic                    fft_sync_r;
logic [2*OW_STFFT-1:0]   fft_result_r, fft_result_rr;

always_ff @(posedge clk_i) begin
    if (reset_i) begin
        fft_sync_r    <= '0;
        fft_result_r  <= '0;
        fft_result_rr <= '0;
    end else begin
        fft_sync_r    <= o_fft_sync;
        fft_result_r  <= o_fft_result;
        fft_result_rr <= fft_result_r;
    end
end

// Split 32-bit bus into two 16-bit signed rails
logic signed [OW_STFFT-1:0] fft_re, fft_im;
assign fft_re = fft_result_rr[2*OW_STFFT-1 : OW_STFFT];  // [31:16]
assign fft_im = fft_result_rr[OW_STFFT-1   : 0];          // [15:0]


// 3. Bin counter and fft_valid
/*
    The R2FFT DMA readout streams all FFT_SIZE bins back-to-back at full
    clock speed — there is no win_ce gating on the output path.
    The counter therefore decrements every clock (not every win_ce).

    We count only N_BINS (129) to discard the conjugate mirror bins
    (130-255) which carry no unique information for a real input signal.

    Timeline relative to o_fft_sync rising:
      clk+0  o_fft_sync=1                 DMA starts inside stfft
      clk+1  fft_sync_r=1  → load 129    first data word at stfft output
      clk+2  bin_cnt=129   fft_result_rr=bin[0]   fft_valid=1
      ...
      clk+130 bin_cnt=1    fft_result_rr=bin[128]  fft_valid=1
      clk+131 bin_cnt=0                             fft_valid=0
*/

localparam int CNT_W = $clog2(N_BINS + 1);
logic [CNT_W-1:0] bin_cnt_q;

always_ff @(posedge clk_i) begin
    if (reset_i)
        bin_cnt_q <= '0;
    else if (fft_sync_r)
        bin_cnt_q <= CNT_W'(N_BINS);
    else if (bin_cnt_q > 0)
        bin_cnt_q <= bin_cnt_q - 1'b1;
end

logic fft_valid;
assign fft_valid = (bin_cnt_q > 0);


// 4. LogMel

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


// 5. Spectrogram buffer

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