module mel_filterbank_new #(
    parameter int N_MELS     = 40,
    parameter int N_BINS     = 129,
    parameter int MAX_COEFFS = 16,
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

    // Flash write for coeff SRAM
    input  logic        flash_coeff_we_i,
    input  logic [9:0]  flash_coeff_addr_i,
    input  logic [15:0] flash_coeff_data_i,

    // Flash write for start bin SRAM
    input  logic        flash_start_we_i,
    input  logic [5:0]  flash_start_addr_i,
    input  logic [7:0]  flash_start_data_i,

    // Flash write for end bin SRAM
    input  logic        flash_end_we_i,
    input  logic [5:0]  flash_end_addr_i,
    input  logic [7:0]  flash_end_data_i
);

    // Power buffer
    logic [POWER_W-1:0] power_buf [N_BINS];
    logic [7:0]         store_ctr;

    // FSM states:
    //
    // STORE:  buffer incoming power bins one per valid_il pulse
    // LOAD:   range_addr presented previous cycle; SRAM reading (1-cycle latency)
    // WAIT1:  start_out/end_out valid; latch them, present first coeff_addr
    // WAIT2:  coeff SRAM reading first weight (1-cycle latency)
    // PROC:   weight valid for current proc_bin, accumulate and present next coeff_addr
    // DRAIN:  last weight arriving, accumulate final product
    // LATCH:  write accum to mel_ol, set up next filter or finish
    typedef enum logic [2:0] { STORE, LOAD, WAIT1, WAIT2, PROC, DRAIN, LATCH } state_t;
    state_t state;

    logic [$clog2(N_MELS)-1:0] mel_idx;
    logic [7:0]                proc_bin;
    logic [7:0]                start_bin_r, end_bin_r;
    logic [ACCUM_W-1:0]        accum;
    logic                      valid_ol_r;

    assign valid_ol = valid_ol_r;

    // SRAM signals
    logic [9:0] coeff_addr;
    logic [5:0] range_addr;
    logic [WEIGHT_W-1:0] weight;
    logic [7:0]          start_out, end_out;

    mel_coeff_sram u_sram (
        .clk_i              (clk_i),
        .flash_coeff_we_i   (flash_coeff_we_i),
        .flash_coeff_addr_i (flash_coeff_addr_i),
        .flash_coeff_data_i (flash_coeff_data_i),
        .flash_start_we_i   (flash_start_we_i),
        .flash_start_addr_i (flash_start_addr_i),
        .flash_start_data_i (flash_start_data_i),
        .flash_end_we_i     (flash_end_we_i),
        .flash_end_addr_i   (flash_end_addr_i),
        .flash_end_data_i   (flash_end_data_i),
        .coeff_addr_i       (coeff_addr),
        .coeff_data_o       (weight),
        .range_addr_i       (range_addr),
        .start_data_o       (start_out),
        .end_data_o         (end_out)
    );

    // Calculate MAC product
    logic [POWER_W+WEIGHT_W-1:0] product;

`ifndef SYNTHESIS
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

    // Each mel filter spans a different bin range (start → end), so filter length varies.
    // Compute cycles scale with filter width: (#bins = end - start + 1).
    // Coefficient offset within current filter (0 to n_coeffs-1)
    logic [7:0] coeff_offset;
    logic [7:0] n_coeffs;     // number of coefficients for current filter

    // Combinational coeff_addr: always derived from mel_idx and coeff_offset
    // This is the address we PRESENT to the SRAM this cycle.
    // The weight we READ this cycle corresponds to the address presented LAST cycle,
    // which is mel_idx * MAX_COEFFS + (coeff_offset - 1) during PROC,
    // or mel_idx * MAX_COEFFS + 0 when entering PROC from WAIT2.

    always_ff @(posedge clk_i) begin
        if (reset_i) begin
            state       <= STORE;
            store_ctr   <= '0;
            mel_idx     <= '0;
            proc_bin    <= '0;
            start_bin_r <= '0;
            end_bin_r   <= '0;
            accum       <= '0;
            valid_ol_r  <= 1'b0;
            range_addr  <= '0;
            coeff_addr  <= '0;
            n_coeffs    <= '0;
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
                            range_addr <= '0;
                            state      <= LOAD;
                        end else begin
                            store_ctr <= store_ctr + 1'b1;
                        end
                    end
                end

                // LOAD: range_addr was presented last cycle.
                //       SRAM is clocking the read, outputs valid next cycle.
                LOAD: begin
                    state <= WAIT1;
                end

                // WAIT1: start_out/end_out now valid.
                //        Latch filter range and Present first coeff_addr to SRAM.
                WAIT1: begin
                    start_bin_r <= start_out;
                    end_bin_r   <= end_out;
                    proc_bin    <= start_out;
                    n_coeffs    <= end_out - start_out + 8'd1;
                    accum       <= '0;
                    coeff_addr  <= mel_idx * 10'(MAX_COEFFS);  // base + 0
                    state       <= WAIT2;
                end

                // WAIT2: coeff SRAM is reading address base+0.
                //        Speculatively present base+1 for the next coefficient.
                //        Weight for base+0 arrives next cycle.
                WAIT2: begin
                    coeff_addr <= coeff_addr + 1'b1;  // present base+1
                    state      <= PROC;
                end

                // PROC: weight for CURRENT proc_bin has arrived (it was
                //       requested 1 cycle ago).  Accumulate product.
                //
                //       Then check if this was the last bin:
                //         proc_bin == end_bin_r → done, go to LATCH
                //         proc_bin + 1 == end_bin_r → one more weight in flight, go to DRAIN
                //         otherwise → advance and stay in PROC
                //
                //       Key invariant: coeff_addr was already incremented
                //       last cycle (in WAIT2 or previous PROC), so the NEXT
                //       weight is already in flight.  We advance proc_bin
                //       to match that in-flight weight.
                PROC: begin
                    accum <= accum + {{(ACCUM_W-POWER_W-WEIGHT_W){1'b0}}, product};

                    if (proc_bin == end_bin_r) begin
                        // Single-bin filter or we just processed the last bin.
                        // No more weights in flight.
                        state <= LATCH;
                    end else if (proc_bin + 8'd1 == end_bin_r) begin
                        // The weight for the LAST bin is already in flight
                        // (coeff_addr was presented last cycle).
                        // Advance proc_bin to match it.  Go to DRAIN.
                        proc_bin <= proc_bin + 1'b1;
                        state    <= DRAIN;
                    end else begin
                        // More bins to go.  Advance proc_bin.
                        // Present next coeff_addr for the bin after that.
                        proc_bin   <= proc_bin + 1'b1;
                        coeff_addr <= coeff_addr + 1'b1;
                    end
                end

                // DRAIN: weight for the last bin (end_bin_r) arrives.
                //        proc_bin == end_bin_r (set in previous PROC cycle).
                //        Accumulate final product.
                DRAIN: begin
                    accum <= accum + {{(ACCUM_W-POWER_W-WEIGHT_W){1'b0}}, product};
                    state <= LATCH;
                end

                // LATCH: write accumulated result, advance to next filter or finish
                LATCH: begin
                    mel_ol[mel_idx] <= accum;

                    if (mel_idx == N_MELS - 1) begin
                        valid_ol_r <= 1'b1;
                        mel_idx    <= '0;
                        state      <= STORE;
                    end else begin
                        mel_idx    <= mel_idx + 1'b1;
                        range_addr <= mel_idx + 1'b1;
                        state      <= LOAD;
                    end
                end

            endcase
        end
    end

endmodule