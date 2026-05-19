module mel_filterbank #(
    parameter int N_MELS     = 40,
    parameter int N_BINS     = 129,
    parameter int MAX_COEFFS = 16,    // max per filter (still used for bounds checking)
    parameter int POWER_W    = 31,
    parameter int WEIGHT_W   = 16,
    parameter int ACCUM_W    = 54
)(
    input  logic [0:0] clk_i,
    input  logic [0:0] reset_i,

    // Data in
    input  logic [POWER_W-1:0] power_il,
    input  logic [0:0]         valid_il,

    // Data out
    output logic [N_MELS-1:0][ACCUM_W-1:0] mel_ol,
    output logic [0:0]                     valid_ol,

    // Flash write — coeff SRAM
    input  logic        flash_coeff_we_i,
    input  logic [7:0]  flash_coeff_addr_i,
    input  logic [15:0] flash_coeff_data_i,

    // Flash write — index SRAM (starts, ends, offsets packed)
    input  logic        flash_index_we_i,
    input  logic [7:0]  flash_index_addr_i,
    input  logic [7:0]  flash_index_data_i,

    // Test mode for SRAM readback
    input  logic        test_mode_i,
    input  logic [7:0]  test_coeff_addr_i,
    input  logic [7:0]  test_index_addr_i
);

    // Power buffer
    logic [POWER_W-1:0] power_buf [N_BINS];
    logic [7:0]         store_ctr;

    // ----------------------------------------------------------------
    // FSM states
    // ----------------------------------------------------------------
    //
    // STORE:   buffer incoming power bins
    // LOAD_S:  present index_addr = mel_idx (start bin address)
    // LOAD_E:  latch start_out, present index_addr = mel_idx + 40 (end bin)
    // LOAD_O:  latch end_out, present index_addr = mel_idx + 80 (coeff offset)
    // WAIT_C:  latch coeff_base, present first coeff_addr
    // WAIT_W:  coeff SRAM reading — weight arrives next cycle
    // PROC:    weight valid — accumulate, advance
    // DRAIN:   final weight arriving — accumulate
    // LATCH:   write result, set up next filter
    typedef enum logic [3:0] {
        STORE, LOAD_S, LOAD_E, LOAD_O, WAIT_C, WAIT_W, PROC, DRAIN, LATCH
    } state_t;
    state_t state;

    logic [$clog2(N_MELS)-1:0] mel_idx;
    logic [7:0]                proc_bin;
    logic [7:0]                start_bin_r, end_bin_r;
    logic [7:0]                coeff_base;   // base offset into sparse coeff array
    logic [ACCUM_W-1:0]        accum;
    logic                      valid_ol_r;

    assign valid_ol = valid_ol_r;

    // SRAM signals
    logic [7:0]  coeff_addr;
    logic [7:0]  index_addr;
    logic [WEIGHT_W-1:0] weight;
    logic [7:0]          index_out;

    mel_coeff_sram u_sram (
        .clk_i              (clk_i),
        .flash_coeff_we_i   (flash_coeff_we_i),
        .flash_coeff_addr_i (flash_coeff_addr_i),
        .flash_coeff_data_i (flash_coeff_data_i),
        .flash_index_we_i   (flash_index_we_i),
        .flash_index_addr_i (flash_index_addr_i),
        .flash_index_data_i (flash_index_data_i),
        .coeff_addr_i       (coeff_addr),
        .coeff_data_o       (weight),
        .index_addr_i       (index_addr),
        .index_data_o       (index_out),
        .test_mode_i        (test_mode_i),
        .test_coeff_addr_i  (test_coeff_addr_i),
        .test_index_addr_i  (test_index_addr_i)
    );

    // Calculate MAC product
    logic [POWER_W+WEIGHT_W-1:0] product;


`ifdef SIM
    assign product = power_buf[proc_bin] * weight;
`else
    MulUns #(
        .widthX(POWER_W),
        .widthY(WEIGHT_W),
        .speed(2)
    ) u_mul (
        .X(power_buf[proc_bin]),
        .Y(weight),
        .P(product)
    );
`endif

    // ----------------------------------------------------------------
    // Pipeline timing
    // ----------------------------------------------------------------
    //
    // Index SRAM has 1-cycle read latency.  We need three reads per
    // filter (start, end, offset), done sequentially:
    //
    //  State  | index_addr presented | index_out valid for
    //  -------+---------------------+---------------------
    //  LOAD_S | mel_idx + 0         | —
    //  LOAD_E | mel_idx + 40        | start_bin  → latch
    //  LOAD_O | mel_idx + 80        | end_bin    → latch
    //  WAIT_C | —                   | coeff_base → latch, present first coeff_addr
    //
    // Coeff SRAM also has 1-cycle latency:
    //
    //  WAIT_C | coeff_addr = base   | —
    //  WAIT_W | coeff_addr = base+1 | (in flight)
    //  PROC   | coeff_addr = base+2 | weight[base+0] arrives → accumulate
    //  PROC   | base+3              | weight[base+1] arrives → accumulate
    //  ...
    //  DRAIN  | —                   | weight[last] arrives   → accumulate
    //  LATCH  | —                   | store accum

    always_ff @(posedge clk_i) begin
        if (reset_i) begin
            state       <= STORE;
            store_ctr   <= '0;
            mel_idx     <= '0;
            proc_bin    <= '0;
            start_bin_r <= '0;
            end_bin_r   <= '0;
            coeff_base  <= '0;
            accum       <= '0;
            valid_ol_r  <= 1'b0;
            index_addr  <= '0;
            coeff_addr  <= '0;
            for (int i = 0; i < N_MELS; i++) mel_ol[i] <= '0;
        end else begin
            valid_ol_r <= 1'b0;

            case (state)

                // STORE: buffer incoming power bins one per valid_il pulse
                STORE: begin
                    if (valid_il) begin
                        power_buf[store_ctr] <= power_il;
                        if (store_ctr == N_BINS - 1) begin
                            store_ctr  <= '0;
                            mel_idx    <= '0;
                            index_addr <= 8'd0;          // present start_bin addr for filter 0
                            state      <= LOAD_S;
                        end else begin
                            store_ctr <= store_ctr + 1'b1;
                        end
                    end
                end

                // LOAD_S: index_addr = mel_idx presented last cycle.
                //         SRAM reading start_bin — arrives next cycle.
                //         Present end_bin address now.
                LOAD_S: begin
                    index_addr <= mel_idx + 8'd40;       // present end_bin addr
                    state      <= LOAD_E;
                end

                // LOAD_E: start_bin has arrived from SRAM. Latch it.
                //         Present offset address.
                LOAD_E: begin
                    start_bin_r <= index_out;             // latch start_bin
                    index_addr  <= mel_idx + 8'd80;      // present offset addr
                    state       <= LOAD_O;
                end

                // LOAD_O: end_bin has arrived. Latch it.
                //         Offset arrives next cycle.
                LOAD_O: begin
                    end_bin_r <= index_out;               // latch end_bin
                    state     <= WAIT_C;
                end

                // WAIT_C: coeff offset has arrived. Latch it.
                //         Set up proc_bin, present first coeff_addr.
                WAIT_C: begin
                    coeff_base <= index_out;              // latch coeff_base
                    proc_bin   <= start_bin_r;
                    accum      <= '0;
                    coeff_addr <= index_out;              // present base+0 to coeff SRAM
                    state      <= WAIT_W;
                end

                // WAIT_W: first coeff_addr presented last cycle, weight in flight.
                //         Speculatively present base+1.
                WAIT_W: begin
                    coeff_addr <= coeff_addr + 1'b1;     // present base+1
                    state      <= PROC;
                end

                // PROC: weight valid for current proc_bin. Accumulate.
                PROC: begin
                    accum <= accum + {{(ACCUM_W-POWER_W-WEIGHT_W){1'b0}}, product};

                    if (proc_bin == end_bin_r) begin
                        // Single-bin or last bin — done
                        state <= LATCH;
                    end else if (proc_bin + 8'd1 == end_bin_r) begin
                        // Last weight already in flight → DRAIN
                        proc_bin <= proc_bin + 1'b1;
                        state    <= DRAIN;
                    end else begin
                        // More bins — advance
                        proc_bin   <= proc_bin + 1'b1;
                        coeff_addr <= coeff_addr + 1'b1;
                    end
                end

                // DRAIN: final weight arrives — accumulate
                DRAIN: begin
                    accum <= accum + {{(ACCUM_W-POWER_W-WEIGHT_W){1'b0}}, product};
                    state <= LATCH;
                end

                // LATCH: store result, set up next filter or finish
                LATCH: begin
                    mel_ol[mel_idx] <= accum;

                    if (mel_idx == N_MELS - 1) begin
                        valid_ol_r <= 1'b1;
                        mel_idx    <= '0;
                        state      <= STORE;
                    end else begin
                        mel_idx    <= mel_idx + 1'b1;
                        index_addr <= mel_idx + 1'b1;    // present start_bin addr for next filter
                        state      <= LOAD_S;
                    end
                end

            endcase
        end
    end

endmodule