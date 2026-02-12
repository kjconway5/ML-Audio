module frame_hop_ctrl #(
    parameter FFT_SIZE = 256,
    parameter HOP_SIZE = 128
)(
    input  wire clk_i,
    input  wire reset_i,
    input  wire ce_i,          // input sample valid

    output wire frame_start_o, // 1-cycle pulse
    output wire frame_ce_o     // enable samples to FFT
);

   
    // Hop counter (counts input samples)
    localparam HOP_MAX = HOP_SIZE - 1;

    logic [$clog2(HOP_SIZE)-1:0] hop_cnt;
    wire hop_wrap = (hop_cnt == HOP_MAX) & ce_i;

    counter #(
        .max_val_p(HOP_MAX)
    ) hop_counter (
        .clk_i   (clk_i),
        .reset_i (reset_i),
        .up_i    (ce_i),
        .down_i  (1'b0),
        .count_o (hop_cnt)
    );


    // Frame counter (counts FFT samples)
    localparam FRAME_MAX = FFT_SIZE - 1;

    logic [$clog2(FFT_SIZE)-1:0] frame_cnt;
    logic frame_active;

    counter #(
        .max_val_p(FRAME_MAX)
    ) frame_counter (
        .clk_i   (clk_i),
        .reset_i (reset_i),
        .up_i    (frame_active & ce_i),
        .down_i  (1'b0),
        .count_o (frame_cnt)
    );



    always_ff @(posedge clk_i) begin
        if (reset_i) begin
            frame_active <= 1'b0;
        end else begin
            // Start frame at hop boundary
            if (hop_wrap)
                frame_active <= 1'b1;

            // End frame after 256 samples
            else if (frame_active && ce_i && frame_cnt == FRAME_MAX)
                frame_active <= 1'b0;
        end
    end

    assign frame_start_o = hop_wrap;
    assign frame_ce_o = frame_active & ce_i;

endmodule
