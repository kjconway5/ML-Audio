module packetizer #(
    parameter N_MELS = 40,
    parameter OUT_W  = 16
)(
    input  logic                clk_i,
    input  logic                reset_i,

    // From pipeline output_buffer
    input  logic [OUT_W-1:0]    data_i,
    input  logic                valid_i,
    output logic                ready_o,

    // To UART TX (AXI-Stream)
    output logic [7:0]          tx_tdata_o,
    output logic                tx_tvalid_o,
    input  logic                tx_tready_i
);

    // Feature buffer
    logic [OUT_W-1:0] feat_buf [N_MELS];
    logic [5:0]       cap_cnt;
    logic [15:0]      frame_id;

    // Packet FSM
    typedef enum logic [1:0] {IDLE, CAPTURE, SEND} state_t;
    state_t state;

    logic [6:0] byte_idx;               // 0..86 within packet
    logic [7:0] checksum;

    // Current byte mux
    logic [7:0] current_byte;
    logic [5:0] mel_idx;
    logic       low_byte;

    always_comb begin
        mel_idx  = (byte_idx - 7'd6) >> 1;
        low_byte = ~byte_idx[0];

        case (byte_idx)
            7'd0:    current_byte = 8'hAA;
            7'd1:    current_byte = 8'h55;
            7'd2:    current_byte = 8'h01;
            7'd3:    current_byte = frame_id[15:8];
            7'd4:    current_byte = frame_id[7:0];
            7'd5:    current_byte = 8'd40;
            7'd86:   current_byte = checksum;
            default: current_byte = low_byte ? feat_buf[mel_idx][7:0]
                                             : feat_buf[mel_idx][15:8];
        endcase
    end

    // Main FSM
    always_ff @(posedge clk_i) begin
        if (reset_i) begin
            state       <= IDLE;
            cap_cnt     <= 0;
            byte_idx    <= 0;
            checksum    <= 0;
            frame_id    <= 0;
            ready_o     <= 1;
            tx_tvalid_o <= 0;
        end else begin
            case (state)

                IDLE: begin
                    ready_o     <= 1;
                    tx_tvalid_o <= 0;
                    cap_cnt     <= 0;
                    if (valid_i) begin
                        feat_buf[0] <= data_i;
                        cap_cnt     <= 1;
                        state       <= CAPTURE;
                    end
                end

                CAPTURE: begin
                    ready_o <= 1;
                    if (valid_i) begin
                        feat_buf[cap_cnt] <= data_i;
                        if (cap_cnt == N_MELS - 1) begin
                            ready_o     <= 0;
                            byte_idx    <= 0;
                            checksum    <= 0;
                            tx_tvalid_o <= 0;
                            state       <= SEND;
                        end else begin
                            cap_cnt <= cap_cnt + 1;
                        end
                    end
                end

                SEND: begin
                    ready_o <= 0;

                    if (!tx_tvalid_o) begin
                        // Present next byte on AXI-Stream
                        tx_tdata_o  <= current_byte;
                        tx_tvalid_o <= 1;
                    end else if (tx_tvalid_o && tx_tready_i) begin
                        // Byte accepted by UART TX
                        checksum <= checksum ^ tx_tdata_o;

                        if (byte_idx == 7'd86) begin
                            tx_tvalid_o <= 0;
                            frame_id    <= frame_id + 1;
                            state       <= IDLE;
                        end else begin
                            byte_idx    <= byte_idx + 1;
                            tx_tvalid_o <= 0;
                        end
                    end
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule
