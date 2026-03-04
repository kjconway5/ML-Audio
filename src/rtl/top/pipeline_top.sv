
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
    input  logic [0:0]          cnn_ready_il  
);


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

    
    logic [OW_STFFT-1:0] fft_re, fft_im;
    assign fft_re = fft_result_rr[2*OW_STFFT-1:OW_STFFT];
    assign fft_im = fft_result_rr[OW_STFFT-1:0];

    
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
        .re_il(fft_re),
        .im_il(fft_im),
        .fft_valid_il(fft_valid),
        .fft_sync_il(fft_sync_r),   // 1-cycle delayed: arrives before the data

        // to CNN
        .cnn_data_ol(cnn_data_ol),
        .cnn_valid_ol(cnn_valid_ol),
        .cnn_ready_il(cnn_ready_il)
    );

endmodule
