module pipeline_top #(
    // fifo params
    parameter [31:0] width_p = 8,
    parameter [31:0] depth_log2_p = 8,

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
    parameter int OUT_W      = 16          // output width to CNN
) (
    input  logic [0:0] clk_i,
    input  logic [0:0] reset_i, 

    // fifo inputs 
    input logic [width_p-1:0] data_i,
    input logic [0:0] valid_i,
    input logic [0:0] ready_i,

    // outputs to ML
    output logic [OUT_W-1:0]    cnn_data_ol,
    output logic [0:0]          cnn_valid_ol,
    input  logic [0:0]          cnn_ready_il  // backpressure from CNN

);

    // order:
    // 1. PDM Decimator (waiting on these for later)
    // 2. FIR Filter (waiting on these for later)
    // 3. FIFO
    // 4. STFFT
    // 5. LogMel Top
    // 6. Output to ML

    logic [width_p-1:0] fifo_o;
    logic [0:0] fifo_valid_o;
    logic [0:0] fifo_ready_o;

    fifo_1r1w #(
        .width_p(width_p),
        .depth_log2_p(depth_log2_p)
    ) fifo (
        .clk_i(clk_i),
        .reset_i(reset_i),
        .data_i(data_i),
        .valid_i(valid_i),
        .ready_i(ready_i),
        .ready_o(fifo_ready_o),
        .valid_o(fifo_valid_o),
        .data_o(fifo_o)
    );

    logic [2*OW_STFFT-1:0] o_fft_result;
    logic [0:0] o_fft_sync;

    stfft #(
        .IW(IW_STFFT),
        .OW(OW_STFFT),
        .FFT_SIZE(FFT_SIZE)
    ) fft (
        .i_clk(clk_i),
        .i_reset(reset_i),
        .i_ce(valid_o), // from FIFO
        .i_sample(fifo_o),
        .o_fft_result(o_fft_result),
        .o_fft_sync(o_fft_sync)
    );

    wire signed [OW_STFFT-1:0] fft_real = o_fft_result[2*OW_STFFT-1:OW_STFFT];
    wire signed [OW_STFFT-1:0] fft_imag = o_fft_result[OW_STFFT-1:0];       

    logmel_top #(
        .IW(IW_LOGMEL),
        .SHIFT(SHIFT),
        .N_MELS(N_MELS),
        .N_BINS(N_BINS),
        .MAX_COEFFS(MAX_COEFFS),
        .POWER_W(POWER_W),
        .WEIGHT_W(WEIGHT_W),
        .ACCUM_W (ACCUM_W),
        .LOG_OUT_W(LOG_OUT_W),
        .OUT_W(OUT_W)
    ) logmel (
        .clk_i(clk_i),
        .reset_i(reset_i),

        // from STFT
        .re_il(fft_real),
        .im_il(fft_imag),
        .fft_valid_il(o_fft_sync),
        .fft_sync_il(o_fft_sync),

        // to CNN
        .cnn_data_ol(cnn_data_ol),
        .cnn_valid_ol(cnn_valid_ol),
        .cnn_ready_il(cnn_ready_il)
    );

endmodule
