module pipeline_top #(
    // stfft params — ZipCPU FFT (active)
    parameter IW_STFFT = 14,
    parameter OW_STFFT = 18,
    parameter FFT_SIZE = 256,

    // R2FFT params (swap in when integrating new FFT)
    // parameter int IW_STFFT  = 16,
    // parameter int OW_STFFT  = 16,

    // logmel params
    parameter int IW_LOGMEL  = OW_STFFT,   // STFT output width
    parameter int SHIFT      = 6,          // power_calc shift
    parameter int N_MELS     = 40,         // mel bins
    parameter int N_BINS     = 129,        // FFT bins
    parameter int MAX_COEFFS = 16,         // sparse ROM depth
    parameter int POWER_W    = 31,         // power_calc output width = 2*IW - SHIFT + 1 - 1
    // R2FFT version:
    // parameter int POWER_W    = 2*IW_LOGMEL - SHIFT, // 26
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
    input  logic [0:0] clk_i,
    input  logic [0:0] reset_i,

    // Audio input
    input logic [IW_STFFT-1:0] data_i,
    input logic [0:0] valid_i,

    // R2FFT version (16-bit signed PCM):
    // input  logic signed [15:0] data_i,
    // input  logic               valid_i,

    // outputs to Spectrogram_SRAM
    // Bank A
    output wire                   sp_a_we,
    output wire [ADDR_W-1:0]      sp_a_waddr,
    output wire signed [7:0]      sp_a_wdata,

    // Bank B
    output wire                   sp_b_we,
    output wire [ADDR_W-1:0]      sp_b_waddr,
    output wire signed [7:0]      sp_b_wdata,

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

);


// ================================================================
// 1. STFFT — ZipCPU FFT (active)
// ================================================================

    logic [2*OW_STFFT-1:0] o_fft_result;
    logic [0:0] o_fft_sync;
    logic       win_ce_raw;   // one pulse per windowed sample entering fftmain

    stfft #(
        .IW(IW_STFFT),
        .OW(OW_STFFT),
        .FFT_SIZE(FFT_SIZE)
    ) fft (
        .i_clk(clk_i),
        .i_reset(reset_i),
        .i_ce(valid_i),
        .i_sample(data_i),
        .o_fft_result(o_fft_result),
        .o_fft_sync(o_fft_sync),
        .win_ce_o(win_ce_raw)
    );

// ================================================================
// 1. STFFT — R2FFT (commented out, integrate later)
// ================================================================
/*
    Timing contract (R2FFT):
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

logic [2*OW_STFFT-1:0] o_fft_result;  // {re[15:0], im[15:0]}
logic                   o_fft_sync;
logic                   win_ce_raw;    // windowed input CE — debug only
logic signed [7:0]      bfpexp_unused;

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
*/


// ================================================================
// 2. FFT output pipeline registers
// ================================================================

    logic               fft_sync_r;
    logic [2*OW_STFFT-1:0] fft_result_r, fft_result_rr;
    logic               win_ce_r, win_ce_rr;

    always_ff @(posedge clk_i) begin
        if (reset_i) begin
            fft_sync_r    <= '0;
            fft_result_r  <= '0;
            fft_result_rr <= '0;
            win_ce_r      <= '0;
            win_ce_rr     <= '0;
        end else begin
            fft_sync_r    <= o_fft_sync;       // 1-cycle delayed sync
            fft_result_r  <= o_fft_result;     // 1-cycle delayed data
            fft_result_rr <= fft_result_r;     // 2-cycle delayed data
            win_ce_r      <= win_ce_raw;       // 1-cycle delayed CE
            win_ce_rr     <= win_ce_r;         // 2-cycle delayed CE (aligned with fft_result_rr)
        end
    end

// R2FFT version (no win_ce pipeline — output streams at full clock):
/*
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
*/

    
    logic [OW_STFFT-1:0] fft_re, fft_im;
    assign fft_re = fft_result_rr[2*OW_STFFT-1:OW_STFFT];
    assign fft_im = fft_result_rr[OW_STFFT-1:0];


// ================================================================
// 3. Bin counter — ZipCPU version (gated by win_ce_rr)
// ================================================================
    
    localparam int CNT_W = $clog2(N_BINS + 1);
    logic [CNT_W-1:0] bin_cnt_q;

    always_ff @(posedge clk_i) begin
        if (reset_i)
            bin_cnt_q <= '0;
        else if (fft_sync_r)                    // load on the pre-data sync pulse
            bin_cnt_q <= CNT_W'(N_BINS);
        else if (bin_cnt_q > 0 && win_ce_rr)   // count only on actual FFT output beats
            bin_cnt_q <= bin_cnt_q - 1'b1;
    end

    logic fft_valid;
    assign fft_valid = (bin_cnt_q > 0) && win_ce_rr;

// ================================================================
// 3. Bin counter — R2FFT version (no win_ce gating, commented out)
// ================================================================
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

always_ff @(posedge clk_i) begin
    if (reset_i)
        bin_cnt_q <= '0;
    else if (fft_sync_r)
        bin_cnt_q <= CNT_W'(N_BINS);
    else if (bin_cnt_q > 0)
        bin_cnt_q <= bin_cnt_q - 1'b1;
end

assign fft_valid = (bin_cnt_q > 0);
*/


// ================================================================
// 4. LogMel
// ================================================================

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
    ) u_logmel (
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


// ================================================================
// 5. Spectrogram buffer
// ================================================================

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