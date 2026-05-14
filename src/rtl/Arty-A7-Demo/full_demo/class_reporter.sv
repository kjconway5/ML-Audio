// class_reporter.sv — Stage 3 result emitter.
//
// On every rising edge of `done` from kws_top, latches `class_out[2:0]`
// and pushes a single tagged byte (0xC0 | class) out the UART TX. The
// tag bits (b7..b3 = 0b11000) let the host distinguish a class result
// from arbitrary streamed data without a full packet framing layer.
//
// TX is one-shot per `done` pulse and yields the TX line otherwise so
// boot_controller ACKs and the spect_streamer remain priority.

module class_reporter (
    input  wire        clk_i,
    input  wire        reset_i,

    input  wire        done_i,
    input  wire [2:0]  class_i,

    output logic [7:0] tx_byte_o,
    output logic       tx_valid_o,
    input  wire        tx_ready_i,

    output logic       busy_o
);

    typedef enum logic [0:0] {
        ST_IDLE,
        ST_TX
    } state_e;

    state_e     state, next_state;
    logic [2:0] class_q;

    // Edge-detect `done` so a held-high `done` only fires once.
    logic done_q;
    wire  done_rise = done_i & ~done_q;
    always_ff @(posedge clk_i) begin
        if (reset_i) done_q <= 1'b0;
        else         done_q <= done_i;
    end

    always_comb begin
        next_state = state;
        case (state)
            ST_IDLE: if (done_rise)    next_state = ST_TX;
            ST_TX:   if (tx_ready_i)   next_state = ST_IDLE;
        endcase
    end

    always_ff @(posedge clk_i) begin
        if (reset_i) begin
            state   <= ST_IDLE;
            class_q <= 3'd0;
        end else begin
            state <= next_state;
            if (state == ST_IDLE && done_rise)
                class_q <= class_i;
        end
    end

    assign tx_byte_o  = {5'b11000, class_q};   // 0xC0 | class
    assign tx_valid_o = (state == ST_TX);
    assign busy_o     = (state != ST_IDLE);

endmodule
