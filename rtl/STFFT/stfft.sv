module stfft #(
    parameter IW = 16,
    parameter OW = 18,
    parameter FFT_SIZE = 256,
    parameter HOP_SIZE = 128 
)(
    input  wire             i_clk,
    input  wire             i_reset,
    input  wire             i_ce,
    input  wire [IW-1:0]    i_sample,
    output wire [2*OW-1:0]  o_fft_result,
    output wire             o_fft_sync
);

/*
Delay Buffer -> Framing -> Windowing -> FFT
*/

    wire [IW-1:0] o_sample;
    logic o_ce, o_frame;

    // Delay buffer
    logic [IW-1:0] buf_sample;
    logic          buf_valid;

    delaybuffer #(
        .width_p(IW),
        .delay_p(FFT_SIZE)
    ) sample_buf (
        .clk_i    (i_clk),
        .reset_i  (i_reset),

        .data_i   (i_sample),
        .valid_i  (i_ce),
        .ready_o  (),          // always ready

        .valid_o  (buf_valid),
        .data_o   (buf_sample),
        .ready_i  (1'b1)
    );


    logic frame_start;
    logic frame_ce;

    frame_hop_ctrl #(
        .FFT_SIZE(FFT_SIZE),
        .HOP_SIZE(HOP_SIZE)
    ) ctrl (
        .clk_i        (i_clk),
        .reset_i      (i_reset),
        .ce_i         (i_ce),
        .frame_start_o(frame_start),
        .frame_ce_o   (frame_ce)
    );



    wire [IW-1:0] win_sample;
    logic win_ce;

    // Windowing
    windowfn #(
        .IW(IW),
        .OW(OW),
        .TW(IW),
        .LGNFFT($clog2(FFT_SIZE))
    ) win (
        .i_clk(i_clk),
        .i_reset(i_reset),
        .i_tap_wr(1'b0),
        .i_tap({IW{1'b0}}),
        .i_ce(frame_ce),
        .i_alt_ce(1'b1),
        .i_sample(buf_sample),
        .o_sample(win_sample),
        .o_ce(win_ce),
        .o_frame(frame_start)
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
