// pipeline_top.sv
//
// Updated for the new single-channel ready/valid stfft.
//
// Frame timing in the new design (T = cycle stfft first asserts o_valid):
//
//   T-1 : stfft_o_valid = 0                              (idle between frames)
//   T   : stfft_o_valid = 1, stfft_o_data = bin 0        <-- frame starts
//         fft_sync_pulse = 1  (combinational rising edge)
//   T+1 : fft_valid_r=1, fft_result_r=bin 0, fft_sync_r=1
//         <-- logmel sees fft_sync_il pulse on this cycle
//         end of cycle: bin_cnt_q <= N_BINS, bfpexp_for_mel <= bfpexp_raw
//   T+2 : fft_result_rr = bin 0, fft_sync_rr = 1, bin_cnt_q = N_BINS
//         fft_valid = (bin_cnt_q > 0) = 1
//         <-- first valid bin reaches logmel
//   ...
//   T+130 : fft_result_rr = bin 128, bin_cnt_q = 1, fft_valid = 1
//   T+131 : fft_valid = 0  (bin_cnt_q hits 0; logmel stops consuming)
//
// Things that changed vs the dual-channel pipeline_top:
//   - stfft instance uses the new ready/valid port set
//     (i_valid/i_data/i_ready, o_valid/o_data/o_ready/o_last, o_bfpexp).
//   - The SYNC_ALIGN_DELAY shift register is gone: it existed to bridge
//     the old DMA-readout pipeline, which doesn't exist in the new FFT
//     (o_valid and o_data are driven by the same output register).
//   - bin_cnt_q loads on fft_sync_r (one cycle before fft_result_rr
//     latches bin 0), so fft_valid is high exactly during the 129 cycles
//     when bins 0..128 sit on fft_result_rr.
//   - o_bfpexp is now meaningful (the old stfft hardcoded it to 0); the
//     BFP-compensation adder downstream finally does real work.

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
// 1. STFFT — new single-channel ready/valid streaming variant
// ==========================================================================

logic                       stfft_o_valid;
logic [2*OW_STFFT-1:0]      stfft_o_data;
logic                       stfft_o_last;
logic                       stfft_i_ready;     // exposed for hierarchical probing
logic signed [7:0]          bfpexp_raw;

stfft #(
    .IW      (IW_STFFT),
    .OW      (OW_STFFT),
    .FFT_SIZE(FFT_SIZE),
    .HOP     (HOP)
) u_stfft (
    .i_clk    (clk_i),
    .i_reset  (reset_i),

    .i_valid  (valid_i),
    .i_data   (data_i),
    .i_ready  (stfft_i_ready),

    .o_valid  (stfft_o_valid),
    .o_data   (stfft_o_data),
    .o_ready  (1'b1),               // never backpressure the FFT output
    .o_last   (stfft_o_last),
    .o_bfpexp (bfpexp_raw)
);

// ==========================================================================
// 2. Frame-start detection — combinational rising edge of stfft_o_valid
// ==========================================================================

logic stfft_o_valid_d;
always_ff @(posedge clk_i) begin
    if (reset_i) stfft_o_valid_d <= 1'b0;
    else         stfft_o_valid_d <= stfft_o_valid;
end

// 1-cycle pulse on the FIRST cycle of each frame's output stream.
wire fft_sync_pulse = stfft_o_valid && !stfft_o_valid_d;

// ==========================================================================
// 3. FFT-output pipeline registers (2 stages, mirrors old timing)
// ==========================================================================

logic                       fft_valid_r,  fft_valid_rr;
logic [2*OW_STFFT-1:0]      fft_result_r, fft_result_rr;
logic                       fft_sync_r,   fft_sync_rr;

always_ff @(posedge clk_i) begin
    if (reset_i) begin
        fft_valid_r   <= 1'b0;
        fft_valid_rr  <= 1'b0;
        fft_result_r  <= '0;
        fft_result_rr <= '0;
        fft_sync_r    <= 1'b0;
        fft_sync_rr   <= 1'b0;
    end else begin
        fft_valid_r   <= stfft_o_valid;
        fft_valid_rr  <= fft_valid_r;
        fft_result_r  <= stfft_o_data;
        fft_result_rr <= fft_result_r;
        fft_sync_r    <= fft_sync_pulse;
        fft_sync_rr   <= fft_sync_r;
    end
end

logic signed [OW_STFFT-1:0] fft_re, fft_im;
assign fft_re = fft_result_rr[2*OW_STFFT-1 : OW_STFFT];
assign fft_im = fft_result_rr[OW_STFFT-1   : 0];

// ==========================================================================
// 4. Bin counter — gate fft_valid to the first N_BINS cycles of each frame
//
// bin_cnt_q loads on fft_sync_r (one cycle before fft_result_rr latches
// bin 0), so fft_valid is high for exactly the 129 cycles when bins
// 0..128 sit on fft_result_rr.
// ==========================================================================

localparam int CNT_W = $clog2(N_BINS + 1);
logic [CNT_W-1:0] bin_cnt_q;

always_ff @(posedge clk_i) begin
    if (reset_i)            bin_cnt_q <= '0;
    else if (fft_sync_r)    bin_cnt_q <= CNT_W'(N_BINS);
    else if (bin_cnt_q > 0) bin_cnt_q <= bin_cnt_q - 1'b1;
end

logic fft_valid;
assign fft_valid = (bin_cnt_q > 0);

// ==========================================================================
// 5. bfpexp latch
//
// bfpexp_raw is combinational from u_stfft.o_bfpexp (which the FFT
// wrapper drives from the per-RAM bfpexp register based on which RAM is
// currently DMA-ing). Latch on fft_sync_r so logmel's compensation sees
// the right exponent throughout the frame.
// ==========================================================================

logic signed [7:0] bfpexp_for_mel;

always_ff @(posedge clk_i) begin
    if (reset_i)         bfpexp_for_mel <= '0;
    else if (fft_sync_r) bfpexp_for_mel <= bfpexp_raw;
end

// ==========================================================================
// 6. LogMel
//
// fft_sync_il fires one cycle before fft_valid_il goes high, matching
// logmel's expected hand-off pattern.
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

// ==========================================================================
// 7. bfpexp compensation
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
// 8. Spectrogram buffer
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