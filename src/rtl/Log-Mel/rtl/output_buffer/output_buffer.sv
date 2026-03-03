module output_buffer #(
    parameter int N_MELS = 40,
    parameter int OUT_W  = 16
)(
    input  logic                    clk,
    input  logic                    reset,

    // log_lut stuff - load the 40 vals at once
    input  logic [N_MELS-1:0][OUT_W-1:0] log_out_i,
    input  logic                    load_i, // pulse high when log_done fires

    // CNN valid/ready handshake
    output logic [OUT_W-1:0]        cnn_data_o,
    output logic                    cnn_valid_o,
    input  logic                    cnn_ready_i,

    // to frame_control
    output logic                    frame_sent_o  // all 40 values accepted by CNN
);

    logic [OUT_W-1:0] buf_q [N_MELS]; // internal register array
    logic [$clog2(N_MELS)-1:0] rd_ptr_q; // which value to send next
    logic active_q; // buffer has data to send

    always_ff @(posedge clk) begin
        if (reset) begin
            rd_ptr_q     <= '0;
            active_q     <= 1'b0;
            frame_sent_o <= 1'b0;
        end else begin
            frame_sent_o <= 1'b0;  // default

            // load all 40 values when log_lut signals done
            if (load_i) begin
                for (int i = 0; i < N_MELS; i++)
                    buf_q[i] <= log_out_i[i];
                rd_ptr_q <= '0;
                active_q <= 1'b1;
            end

            // send one value per cycle when CNN is ready
            if (active_q && cnn_ready_i) begin
                if (rd_ptr_q == N_MELS-1) begin
                    // last value sent
                    active_q     <= 1'b0;
                    frame_sent_o <= 1'b1;
                end else begin
                    rd_ptr_q <= rd_ptr_q + 1'b1;
                end
            end
        end
    end

    assign cnn_data_o  = buf_q[rd_ptr_q];
    assign cnn_valid_o = active_q;

endmodule
