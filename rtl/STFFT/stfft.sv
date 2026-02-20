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
    output wire             o_fft_sync
);

	reg		alt_ce;
	reg	[4:0]	alt_countdown;

    initial	alt_countdown = 0;

	always @(posedge i_clk) begin
	    if (i_reset) begin
		    alt_ce <= 1'b0;
		    alt_countdown <= 5'd22;
	    end else if (i_ce) begin
		    alt_countdown <= 5'd22;
		    alt_ce <= 1'b0;
	    end else if (alt_countdown > 0) begin
		    alt_countdown <= alt_countdown - 1'b1;
		    alt_ce <= (alt_countdown <= 1);
	    end else
		    alt_ce <= 1'b0;
    end

    wire [IW-1:0]    win_sample;
    wire win_ce;

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
        .i_ce(i_ce),
        .i_alt_ce(i_alt_ce),
        .i_sample(i_sample),
        .o_sample(win_sample),
        .o_ce(win_ce),
        .o_frame()
    );

    // FFT
    fftmain fft (
        .i_clk(i_clk),
        .i_reset(i_reset),
        .i_ce(win_ce),
        .i_sample({win_sample, {IW{1'b0}}}), // real + 0 imaginary
        .o_result(o_fft_result),
        .o_sync(o_fft_sync)
    );



endmodule
