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
    parameter int OUT_W      = 16          // output width to CNN
) (
    input  logic [0:0] clk_i,
    input  logic [0:0] reset_i,

    // Audio input
    input logic [IW_STFFT-1:0] data_i,
    input logic [0:0] valid_i,

    // outputs to ML
    output logic [OUT_W-1:0]    cnn_data_ol,
    output logic [0:0]          cnn_valid_ol,
    input  logic [0:0]          cnn_ready_il  // backpressure from CNN

);

    // order:
    // 1. PDM Decimator (waiting on these for later)
    // 2. FIR Filter (waiting on these for later)
    // 3. STFFT
    // 4. LogMel Top
    // 5. Output to ML

    logic [2*OW_STFFT-1:0] o_fft_result;
    logic [0:0] o_fft_sync;

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
        .o_fft_sync(o_fft_sync)
    );


    // Timing bridge
    logic               fft_sync_r;
    logic [2*OW_STFFT-1:0] fft_result_r, fft_result_rr;

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

    // Split 2-cycle-delayed result into real and imaginary
    wire signed [OW_STFFT-1:0] fft_real = fft_result_rr[2*OW_STFFT-1:OW_STFFT];
    wire signed [OW_STFFT-1:0] fft_imag = fft_result_rr[OW_STFFT-1:0];


    // Asserts fft_valid for exactly N_BINS cycles after sync
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

        // from STFT (timing-bridged)
        .re_il(fft_real),
        .im_il(fft_imag),
        .fft_valid_il(fft_valid),
        .fft_sync_il(fft_sync_r),

        // to CNN
        .cnn_data_ol(cnn_data_ol),
        .cnn_valid_ol(cnn_valid_ol),
        .cnn_ready_il(cnn_ready_il)
    );

endmodule