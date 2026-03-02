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

    wire [IW-1:0]    win_sample;
    wire win_ce;
	
	logic phase_d, phase_q;

	always @(posedge i_clk) begin
		if (i_reset) begin
	    	phase_q <= 1'b0;
		end else begin
	    	phase_q <= phase_d;
		end
	end

	always_comb begin
		phase_d = phase_q;
		if (i_ce) begin
			phase_d = ~phase_q;
		end else begin
			phase_d = phase_q;
		end
	end
	
	wire primary_ce  =  i_ce & ~phase_q;
	wire alternate_ce = i_ce &  phase_q;

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
