module stfft #(
    parameter IW = 16,
    parameter OW = 18,
    parameter FFT_SIZE = 256,
    parameter HOP_SIZE = 128  // 50% overlap
)(
    input  wire             i_clk,
    input  wire             i_reset,
    input  wire             i_ce,
    input  wire [IW-1:0]    i_sample,
    output wire [2*OW-1:0]  o_fft_result,
    output wire             o_fft_sync
);

    wire [IW-1:0] o_sample;
    logic o_ce, o_frame;

    // Delay buffer
    logic [IW-1:0] buf_sample;
    logic          buf_valid;

    delaybuffer #(
        .width_p(IW),
        .delay_p(256)
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

    // Frame / hop controller
    logic [7:0] sample_cnt;

    always_ff @(posedge clk) begin
        if (reset) begin
        sample_cnt <= 0;
        end else if (i_ce) begin
        if (sample_cnt == 8'd255)
            sample_cnt <= 8'd128;   // hop size
        else
            sample_cnt <= sample_cnt + 1'b1;
        end
    end

    assign o_frame  = (sample_cnt == 0) & i_ce;
    assign o_ce     = buf_valid;
    assign o_sample = buf_sample;


    wire [IW-1:0] win_sample;
    logic          win_ce, frame_start;

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
        .i_ce(frame_ready),
        .i_alt_ce(1'b1),
        .i_sample(frame_buf[write_ptr]),
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
