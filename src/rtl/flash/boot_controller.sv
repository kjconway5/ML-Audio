// boot_controller.sv  — Increment 1: packet framing
//
// What this version does:
//   - Receives bytes from UART RX
//   - Hunts for sync header (0xAA 0x55)
//   - Parses packet header (target, addr_hi, addr_lo, len_hi, len_lo)
//   - Counts payload bytes, computes XOR checksum over header+payload
//   - Validates checksum byte
//   - Sends ACK / NACK / ERR over UART TX
//   - Pulses pkt_valid_o for one cycle when a good packet completes
// Packet format (reminder):
//   byte 0:    0xAA              ─┐ sync header
//   byte 1:    0x55              ─┘
//   byte 2:    target            (high nibble: module, low nibble: subtarget)
//   byte 3:    address[15:8]
//   byte 4:    address[7:0]
//   byte 5:    length[15:8]      (number of payload bytes)
//   byte 6:    length[7:0]
//   bytes 7..: payload
//   last byte: XOR checksum (over bytes 2 through end-of-payload)

module boot_controller
    import boot_pkg::*;
(
    input  logic clk_i,
    input  logic reset_i,

    //UART RX 
    input  logic [7:0] rx_byte_i,
    input  logic rx_valid_i,
    output logic rx_ready_o,

    //UART TX 
    output logic [7:0]        tx_byte_o,
    output logic              tx_valid_o,
    input  logic              tx_ready_i,
    //
    output boot_bus_t         features_boot_o,
    output boot_bus_t         dscnn_boot_o,
    output boot_bus_t         audio_boot_o,   // MOD_AUDIO: streamed 16-bit PCM samples

    // Status
    output logic              boot_done_o,
    output logic              session_reset_o, // 1-cycle pulse on CTRL_SESSION_RESET
    output logic              pkt_valid_o,    // 1-cycle pulse on good packet
    output logic [7:0]        last_target_o,  // observability for debug
    output logic [15:0]       last_addr_o,
    output logic [15:0]       last_len_o
);

    //  FSM state
    typedef enum logic [3:0] {
        S_HUNT_AA,      // wait for first sync byte
        S_HUNT_55,      // wait for second sync byte
        S_RD_TARGET,
        S_RD_ADDR_HI,
        S_RD_ADDR_LO,
        S_RD_LEN_HI,
        S_RD_LEN_LO,
        S_RD_PAYLOAD,   // count bytes, accumulate checksum
        S_RD_CKSUM,     // last byte = host's checksum
        S_SEND_ACK,
        S_SEND_NACK,
        S_SEND_ERR
    } state_e;

    state_e state, next_state;

    //  Latched packet fields

    logic [7:0]  target_q;
    logic [15:0] addr_q;
    logic [15:0] len_q;
    logic [15:0] payload_cnt_q;     // bytes received so far in payload
    logic [7:0]  cksum_acc_q;       // running XOR of header+payload bytes
    
    //Bus Routing
    
    logic        wide_target_q;        // 1 = 16-bit target, 0 = 8-bit
    logic [15:0] write_addr_q;         // auto-incrementing SRAM write address
    logic [7:0]  lo_byte_q;            // holding register for first byte of 16-bit pair
    logic        have_lo_q;            // flag: lo_byte_q has valid data

    //  Always ready to accept bytes
    //  The FSM advances on every handshake.
    assign rx_ready_o = 1'b1;

    wire byte_in = rx_valid_i & rx_ready_o;



    //  Bus write pulse logic
 
    logic        bus_fire;      // issue a bus write this cycle
    logic [15:0] bus_data;      // data to write
 
    always_comb begin
        bus_fire = 1'b0;
        bus_data = 16'h0000;
 
        if (state == S_RD_PAYLOAD && byte_in) begin
            if (!wide_target_q) begin
                // 8-bit target: every byte fires a write
                bus_fire = 1'b1;
                bus_data = {8'h00, rx_byte_i};
            end else begin
                // 16-bit target: fire on hi byte (second of pair)
                if (have_lo_q) begin
                    bus_fire = 1'b1;
                    bus_data = {rx_byte_i, lo_byte_q};
                end
                // else: storing lo byte, no fire yet
            end
        end
    end

    //  Sequential: state, latched fields, checksum accumulator

    always_ff @(posedge clk_i) begin
        if (reset_i) begin
            state         <= S_HUNT_AA;
            target_q      <= 8'h00;
            addr_q        <= 16'h0000;
            len_q         <= 16'h0000;
            payload_cnt_q <= 16'h0000;
            cksum_acc_q   <= 8'h00;
            
            wide_target_q <= 1'b0;
            write_addr_q  <= 16'h0000;
            lo_byte_q     <= 8'h00;
            have_lo_q     <= 1'b0;
            boot_done_o   <= 1'b0;
            session_reset_o <= 1'b0;
            features_boot_o <= '0;
            dscnn_boot_o    <= '0;
            audio_boot_o    <= '0;
        end else begin

            state <= next_state;

            //deassert bus valid + pulses every cycle (1-cycle defaults)
            features_boot_o.valid <= 1'b0;
            dscnn_boot_o.valid    <= 1'b0;
            audio_boot_o.valid    <= 1'b0;
            session_reset_o       <= 1'b0;
            case (state)
                S_HUNT_AA, S_HUNT_55: begin
                    //RESET CHECKSUM 
                    cksum_acc_q   <= 8'h00;
                    payload_cnt_q <= 16'h0000;
                    have_lo_q     <= 1'b0;
                end
                //LATCH  BYTES TO THEIR APPROPRIATE WIRES then CHECK SUM 
                S_RD_TARGET: begin
                    if (byte_in) begin
                        target_q    <= rx_byte_i;
                        cksum_acc_q <= cksum_acc_q ^ rx_byte_i;
                        wide_target_q <= is_target_16bit(rx_byte_i);
                    end
                end

                S_RD_ADDR_HI: begin
                    if (byte_in) begin
                        addr_q[15:8] <= rx_byte_i;
                        cksum_acc_q  <= cksum_acc_q ^ rx_byte_i;
                    end
                end

                S_RD_ADDR_LO: begin
                    if (byte_in) begin
                        addr_q[7:0] <= rx_byte_i;
                        cksum_acc_q <= cksum_acc_q ^ rx_byte_i;
                    end
                end

                S_RD_LEN_HI: begin
                    if (byte_in) begin
                        len_q[15:8] <= rx_byte_i;
                        cksum_acc_q <= cksum_acc_q ^ rx_byte_i;
                    end
                end

                S_RD_LEN_LO: begin
                    if (byte_in) begin
                        len_q[7:0]  <= rx_byte_i;
                        cksum_acc_q <= cksum_acc_q ^ rx_byte_i;
                        // Initialize write address from packet start address
                        // addr_q[15:8] was latched earlier; addr_q[7:0] was
                        // latched in the previous state
                        write_addr_q <= addr_q;
                        have_lo_q    <= 1'b0;
                    end
                end

                S_RD_PAYLOAD: begin
                    if (byte_in) begin
                        payload_cnt_q <= payload_cnt_q + 16'd1;
                        cksum_acc_q   <= cksum_acc_q ^ rx_byte_i;
 
                        if (!wide_target_q) begin
                            // 8-bit target: write every byte
                            // bus_fire and bus_data are set combinationally
                            write_addr_q <= write_addr_q + 16'd1;
                        end else begin
                            // 16-bit target: pair bytes
                            if (!have_lo_q) begin
                                lo_byte_q <= rx_byte_i;
                                have_lo_q <= 1'b1;
                            end else begin
                                // hi byte arrived: bus_fire is set
                                have_lo_q    <= 1'b0;
                                write_addr_q <= write_addr_q + 16'd1;
                            end
                        end
 
                        //Drive the appropriate boot bus
                        if (bus_fire) begin
                            case (target_q[7:4])
                                MOD_FEATURES: begin
                                    features_boot_o.valid     <= 1'b1;
                                    features_boot_o.subtarget <= target_q[3:0];
                                    features_boot_o.addr      <= write_addr_q;
                                    features_boot_o.data      <= bus_data;
                                end
                                MOD_DSCNN: begin
                                    dscnn_boot_o.valid     <= 1'b1;
                                    dscnn_boot_o.subtarget <= target_q[3:0];
                                    dscnn_boot_o.addr      <= write_addr_q;
                                    dscnn_boot_o.data      <= bus_data;
                                end
                                MOD_AUDIO: begin
                                    // 16-bit packed PCM samples streamed to pipeline_top.
                                    // subtarget unused (always 0); addr increments so
                                    // downstream can sanity-check sample ordering.
                                    audio_boot_o.valid     <= 1'b1;
                                    audio_boot_o.subtarget <= target_q[3:0];
                                    audio_boot_o.addr      <= write_addr_q;
                                    audio_boot_o.data      <= bus_data;
                                end
                                default: ; // MOD_CONTROL, MOD_DEBUG: no bus write
                            endcase
                        end
                    end
                end
 
                S_SEND_ACK: begin
                    // ── Handle MOD_CONTROL on successful packet ───
                    if (tx_valid_o && tx_ready_i) begin
                        if (target_q[7:4] == MOD_CONTROL) begin
                            case (target_q[3:0])
                                CTRL_BOOT_DONE:     boot_done_o     <= 1'b1;
                                CTRL_SESSION_RESET: session_reset_o <= 1'b1; // 1-cycle pulse
                                default: ;
                            endcase
                        end
                    end
                end

                default: ; // no-op
            endcase
        end
    end

    //next_state
    always_comb begin
        next_state = state;
        //IF FIRST BYTE == AA moves on to next state
        case (state)
            S_HUNT_AA:
                if (byte_in && rx_byte_i == SYNC_BYTE_0)
                    next_state = S_HUNT_55;
        //IF NEXT BYTE == 55 move on to next state, else go back to previous state
            S_HUNT_55:
                if (byte_in) begin
                    if (rx_byte_i == SYNC_BYTE_1)
                        next_state = S_RD_TARGET;
                    else if (rx_byte_i == SYNC_BYTE_0)
                        next_state = S_HUNT_55;       // allow back-to-back AAs
                    else
                        next_state = S_HUNT_AA;       // false alarm
                end

            S_RD_TARGET:  if (byte_in) next_state = S_RD_ADDR_HI;
            S_RD_ADDR_HI: if (byte_in) next_state = S_RD_ADDR_LO;
            S_RD_ADDR_LO: if (byte_in) next_state = S_RD_LEN_HI;
            S_RD_LEN_HI:  if (byte_in) next_state = S_RD_LEN_LO;

            S_RD_LEN_LO:
                if (byte_in) begin
                    // If length is 0, skip payload entirely
                    if (len_q[15:8] == 8'h00 && rx_byte_i == 8'h00)
                        next_state = S_RD_CKSUM;
                    else
                        next_state = S_RD_PAYLOAD;
                end

            S_RD_PAYLOAD:
                // payload_cnt_q is updated NEXT cycle, so check against (cnt+1)
                if (byte_in && (payload_cnt_q + 16'd1 == len_q))
                    next_state = S_RD_CKSUM;
            //CHECK SUM 
            S_RD_CKSUM:
                if (byte_in) begin
                    if (rx_byte_i == cksum_acc_q)
                        next_state = S_SEND_ACK;
                    else
                        next_state = S_SEND_NACK;
                end

            S_SEND_ACK, S_SEND_NACK, S_SEND_ERR:
                if (tx_valid_o && tx_ready_i)
                    next_state = S_HUNT_AA;

            default:
                next_state = S_HUNT_AA;
        endcase
    end


    //  TX byte and valid
    always_comb begin
        tx_byte_o  = 8'h00;
        tx_valid_o = 1'b0;

        case (state)
            S_SEND_ACK:  begin tx_byte_o = ACK_BYTE;  tx_valid_o = 1'b1; end
            S_SEND_NACK: begin tx_byte_o = NACK_BYTE; tx_valid_o = 1'b1; end
            S_SEND_ERR:  begin tx_byte_o = ERR_BYTE;  tx_valid_o = 1'b1; end
            default: ;
        endcase
    end
    //  Status outputs (debug)

    assign pkt_valid_o   = (state == S_SEND_ACK) && tx_ready_i && tx_valid_o;
    assign last_target_o = target_q;
    assign last_addr_o   = addr_q;
    assign last_len_o    = len_q;

endmodule