module stfft #(
    parameter IW = 14,
    parameter OW = 18,
    parameter FFT_SIZE = 256
)
(
    input  wire             i_clk,
    input  wire             i_reset,
    input  wire             i_ce,
    input  wire [IW-1:0]    i_sample,
    output wire [2*OW-1:0]  o_fft_result,
    output wire             o_fft_sync,
    output wire             win_ce_o
);

    wire [IW-1:0]    win_sample;
    wire win_ce;

    // ── CE generation ──────────────────────────────────────────────────────
    // primary_ce = i_ce  → fires on every input sample (writes data to windowfn)
    // alternate_ce       → fires 3 clocks after i_ce (overlap processing)
    //
    // Both fire for EVERY input sample so all samples are stored and the
    // 50% overlap produces 2 windowed frames per 256 input samples.
    // Spacing of 3 clocks satisfies fftmain's CKPCE=3 constraint.
    // Requires CE_EVERY >= 6 so alt_ce doesn't collide with the next i_ce.
    wire primary_ce = i_ce;

    reg [2:0] alt_delay;
    always @(posedge i_clk) begin
        if (i_reset)
            alt_delay <= 3'b0;
        else
            alt_delay <= {alt_delay[1:0], i_ce};
    end
    wire alternate_ce = alt_delay[2];

    // Windowing
    windowfn #(
        .IW(IW),
        .OW(IW),
        .TW(IW),
        .LGNFFT($clog2(FFT_SIZE)),
        .INITIAL_COEFFS("hanning.hex")
    ) win (
        .i_clk(i_clk),
        .i_reset(i_reset),
        .i_tap_wr(1'b0),
        .i_tap({IW{1'b0}}),
        .i_ce(primary_ce),
        .i_alt_ce(alternate_ce),
        .i_sample(i_sample),
        .o_sample(win_sample),
        .o_ce(win_ce),
        /* verilator lint_off PINCONNECTEMPTY */
        .o_frame()
        /* verilator lint_on PINCONNECTEMPTY */
    );

    assign win_ce_o = win_ce;

    // ── Width adaptation for fftmain (hardcoded IWIDTH=16, OWIDTH=16) ────
    // fftmain is a generated 256-point FFT with fixed 16-bit I/O (32-bit bus).
    // Input:  sign-extend IW-bit windowed sample → 16-bit real + 16-bit zero imag
    // Output: sign-extend 16-bit real/imag → OW-bit for downstream pipeline
    localparam FFT_IW = 16;   // fftmain's hardcoded IWIDTH
    localparam FFT_OW = 16;   // fftmain's hardcoded OWIDTH

    // Input: sign-extend IW-bit windowed sample to 16-bit real, zero imaginary
    wire signed [FFT_IW-1:0] fft_in_re;
    assign fft_in_re = {{(FFT_IW-IW){win_sample[IW-1]}}, win_sample};
    wire [2*FFT_IW-1:0] fft_in = {fft_in_re, {FFT_IW{1'b0}}};

    // FFT core
    wire [2*FFT_OW-1:0] fft_out;
    wire                 fft_sync_raw;

    fftmain fft (
        .i_clk(i_clk),
        .i_reset(i_reset),
        .i_ce(win_ce),
        .i_sample(fft_in),
        .o_result(fft_out),
        .o_sync(fft_sync_raw)
    );

    // Output: sign-extend 16-bit real/imag to OW-bit
    wire signed [FFT_OW-1:0] fft_out_re = fft_out[2*FFT_OW-1:FFT_OW];
    wire signed [FFT_OW-1:0] fft_out_im = fft_out[FFT_OW-1:0];

    assign o_fft_result = {{(OW-FFT_OW){fft_out_re[FFT_OW-1]}}, fft_out_re,
                           {(OW-FFT_OW){fft_out_im[FFT_OW-1]}}, fft_out_im};
    assign o_fft_sync   = fft_sync_raw;

endmodule
