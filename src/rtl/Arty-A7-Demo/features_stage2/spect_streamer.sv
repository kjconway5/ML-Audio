// spect_streamer.sv — Stage 2 read-back unit.
//
// Same shape as Stage 1.2's verify_streamer, but targeted at the
// spectrogram SRAM (2 banks A/B) inside spect_capture. Snoops
// boot_controller's parsed packet fields; if the completed packet was
// MOD_DEBUG.DBG_READ_SPECT_{A,B}, walks the requested bank and streams
// `last_len` bytes out the UART TX.
//
// The host packs `len` zero bytes of dummy payload so the existing
// boot_controller framing/checksum FSM is unchanged.

module spect_streamer
    import boot_pkg::*;
#(
    parameter int ADDR_W = 11
) (
    input  wire                clk_i,
    input  wire                reset_i,

    // Snoop from boot_controller
    input  wire                pkt_valid_i,
    input  wire [7:0]          last_target_i,
    input  wire [15:0]         last_addr_i,
    input  wire [15:0]         last_len_i,

    // Spect SRAM read port
    output logic [ADDR_W-1:0]  spect_raddr_o,
    output logic               spect_bank_sel_o,   // 0 = A, 1 = B
    input  wire  signed [7:0]  spect_rdata_i,

    // UART TX (muxed externally with boot_controller TX)
    output logic [7:0]         tx_byte_o,
    output logic               tx_valid_o,
    input  wire                tx_ready_i,

    output logic               busy_o
);

    typedef enum logic [1:0] {
        ST_IDLE,
        ST_SETUP,   // drive addr
        ST_WAIT,    // SRAM read latency
        ST_TX       // hold tx_valid until tx_ready
    } state_e;

    state_e      state, next_state;

    logic        bank_q;
    logic [15:0] byte_cnt_q;
    logic [15:0] byte_len_q;
    logic [15:0] byte_addr_q;

    wire is_mod_debug = (last_target_i[7:4] == MOD_DEBUG);
    wire is_spect_a   = (last_target_i[3:0] == DBG_READ_SPECT_A);
    wire is_spect_b   = (last_target_i[3:0] == DBG_READ_SPECT_B);
    wire trigger      = pkt_valid_i & is_mod_debug & (is_spect_a | is_spect_b)
                                    & (last_len_i != 16'd0);

    assign spect_raddr_o    = byte_addr_q[ADDR_W-1:0];
    assign spect_bank_sel_o = bank_q;

    always_comb begin
        next_state = state;
        case (state)
            ST_IDLE:  if (trigger)              next_state = ST_SETUP;
            ST_SETUP:                           next_state = ST_WAIT;
            ST_WAIT:                            next_state = ST_TX;
            ST_TX:    if (tx_ready_i) begin
                if (byte_cnt_q + 16'd1 == byte_len_q)
                                                next_state = ST_IDLE;
                else
                                                next_state = ST_SETUP;
            end
        endcase
    end

    always_ff @(posedge clk_i) begin
        if (reset_i) begin
            state       <= ST_IDLE;
            bank_q      <= 1'b0;
            byte_cnt_q  <= 16'd0;
            byte_len_q  <= 16'd0;
            byte_addr_q <= 16'd0;
        end else begin
            state <= next_state;

            if (state == ST_IDLE && trigger) begin
                bank_q      <= is_spect_b;
                byte_addr_q <= last_addr_i;
                byte_len_q  <= last_len_i;
                byte_cnt_q  <= 16'd0;
            end else if (state == ST_TX && tx_ready_i) begin
                byte_cnt_q  <= byte_cnt_q  + 16'd1;
                byte_addr_q <= byte_addr_q + 16'd1;
            end
        end
    end

    assign tx_byte_o  = spect_rdata_i;
    assign tx_valid_o = (state == ST_TX);
    assign busy_o     = (state != ST_IDLE);

endmodule
