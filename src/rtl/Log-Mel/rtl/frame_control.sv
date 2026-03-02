module frame_control #(
    parameter int MEL_BINS = 40
) (
    input logic clk,
    input logic reset,

    // inputs from other modules
    input logic fft_sync_i, // o_sync from stft, when HI marks first bin of new frame
    input logic frame_sent_i, // all 40 values taken by CNN from output buffer
    input logic filterbank_done_i, // valid_ol from filterbank

    // outputs to log_lut
    output logic [$clog2(MEL_BINS)-1:0] mel_idx_o, // which mel bin to compress (0-39)
    output logic log_en_o, // high during LOG_COMPRESS state
    
    // outputs to output_buffer
    output logic output_valid_o // high during OUTPUT state when buffer has valid data
);

    // mel counter
    logic [$clog2(MEL_BINS)-1:0] mel_idx_l;

    always_ff @(posedge clk) begin
        if (reset) begin
            curr_state_q <= IDLE;
            mel_idx_l <= '0;
        end else begin
            curr_state_q <= next_state_d;

            // increment mel_idx during LOG_COMPRESS
            if (curr_state_q == LOG_COMPRESS)
                mel_idx_l <= mel_idx_l + 1'b1;
            else
                mel_idx_l <= '0;
        end
    end

    typedef enum logic[1:0] {
        IDLE = 2'd0, // wait for new frame from STFT
        ACCUMULATE = 2'd1, // stream 128 bins thru MACs in mel filterbank
        LOG_COMPRESS = 2'd2, // one by one compress mel energies through log_lut
        OUTPUT = 2'd3 // out_buffer sends values to CNN
    } FRAME_CTRL_STATE;

    FRAME_CTRL_STATE curr_state_q, next_state_d;

    always_ff @(posedge clk) begin
        if (reset) begin
            curr_state_q <= IDLE;

        end else begin
            curr_state_q <= next_state_d;

        end
    end

    always_comb begin : frame_handling
        next_state_d = curr_state_q;

        unique case(curr_state_q)
            IDLE: begin
                if (fft_sync_i) begin
                    next_state_d = ACCUMULATE;
                end else begin
                    next_state_d = IDLE;
                end
            end

            ACCUMULATE: begin
                if (bin_counter_o == (filterbank_done_i-1)) begin
                    next_state_d = LOG_COMPRESS;
                end else begin
                    next_state_d = ACCUMULATE;
                end
            end

            LOG_COMPRESS: begin
                if (mel_idx_o == (MEL_BINS-1)) begin
                    next_state_d = OUTPUT;
                end else begin
                    next_state_d = LOG_COMPRESS;
                end
            end

            OUTPUT: begin
                if (frame_sent_i) begin
                    next_state_d = IDLE;
                end else begin
                    next_state_d = OUTPUT;
                end
            end 
        endcase
    end

    // output assignments
    assign mel_idx_o = mel_idx_l;
    assign log_en_o = (curr_state_q == LOG_COMPRESS);
    assign output_valid_o = (curr_state_q == OUTPUT);

    endmodule