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

    // Flash write for start/end indice SRAM 
    input  logic        flash_index_we_i,
    input  logic [6:0]  flash_index_addr_i,
    input  logic [7:0]  flash_index_data_i
);


    // Power buffer
    logic [POWER_W-1:0] power_buf [N_BINS];
    logic [7:0]         store_ctr;


    // FSM State enum
    typedef enum logic [1:0] { STORE, LOAD, PROC, LATCH } state_t;
    state_t state;

    logic [$clog2(N_MELS)-1:0] mel_idx;
    logic [7:0]                proc_bin;
    logic [7:0]                start_bin_r, end_bin_r;
    logic [ACCUM_W-1:0]        accum;
    logic                      valid_ol_r;

    assign valid_ol = valid_ol_r;


    // SRAM address signals
    logic [9:0] coeff_addr;
    logic [6:0] index_addr;
    logic [WEIGHT_W-1:0] weight;
    logic [7:0]          index_out;

    // calculate coeff address: mel_idx * MAX_COEFFS + offset into filter
    assign coeff_addr = mel_idx * MAX_COEFFS + (proc_bin - start_bin_r);

    // start/end bin index address: mel_idx for start, mel_idx + N_MELS for end
    // driven by FSM below

    mel_coeff_sram u_sram (
        .clk_i               (clk_i),
        .flash_coeff_we_i    (flash_coeff_we_i),
        .flash_coeff_addr_i  (flash_coeff_addr_i),
        .flash_coeff_data_i  (flash_coeff_data_i),
        .flash_index_we_i     (flash_index_we_i),
        .flash_index_addr_i   (flash_index_addr_i),
        .flash_index_data_i   (flash_index_data_i),
        .coeff_addr_i        (coeff_addr),
        .coeff_data_o        (weight),
        .index_addr_i         (index_addr),
        .index_data_o         (index_out)
    );


    // Delay pipeline by 1 cycle to match SRAM read latency
    // Weight arrives 1 cycle after coeff_addr calculated
    // Delay proc_bin and end_bin_r to match
    logic [7:0] proc_bin_d;
    logic [7:0] end_bin_d;
    logic [0:0] proc_en;  // high in PROC state
    logic [0:0] proc_en_d;

    always_ff @(posedge clk_i) begin
        if (reset_i) begin
            proc_bin_d <= '0;
            end_bin_d  <= '0;
            proc_en_d  <= 1'b0;
        end else begin
            proc_bin_d <= proc_bin;
            end_bin_d  <= end_bin_r;
            proc_en_d  <= proc_en;
        end
    end


    // Calculate power values by corresponding filter weight
    logic [POWER_W+WEIGHT_W-1:0] product;

`ifndef SYNTHESIS
    assign product = power_buf[proc_bin_d] * weight;
`else
    MulUns #(
        .widthX(POWER_W),
        .widthY(WEIGHT_W),
        .speed(2)
    ) u_mul (
        .X(power_buf[proc_bin_d]),
        .Y(weight),
        .P(product)
    );
`endif

    // LOAD now takes 2 cycles: one to read start, one to read end
    // PROC fetches coeff_addr and accumulate happens next cycle with the 1 cycle delay
    logic load_phase;  // 0=reading start, 1=reading end

    assign proc_en = (state == PROC);

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
            load_phase  <= 1'b0;
            index_addr   <= '0;
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
                            load_phase <= 1'b0;
                            index_addr  <= '0;  // request start for filter 0
                            state      <= LOAD;
                        end else begin
                            store_ctr <= store_ctr + 1'b1;
                        end
                    end
                end

                // Cycle 0: index_addr = mel_idx to get start_bin next cycle
                // Cycle 1: index_addr = mel_idx+N_MELS to get end_bin next cycle
                // Two cycles total, then into PROC
                LOAD: begin
                    if (!load_phase) begin
                        // Cycle 0 result: start_bin now valid from previous request
                        // (first entry: index_addr was set before entering LOAD)
                        start_bin_r <= index_out;
                        index_addr   <= mel_idx + N_MELS[6:0];  // request end bin
                        load_phase  <= 1'b1;
                    end else begin
                        // Cycle 1 result: end_bin now valid
                        end_bin_r  <= index_out;
                        proc_bin   <= start_bin_r;
                        load_phase <= 1'b0;
                        accum      <= '0;
                        state      <= PROC;
                    end
                end

                // Fetch coeff_addr this cycle
                // Accumulates weight next cycle
                // Also accumulate the result from the previous PROC cycle.
                PROC: begin
                    // Accumulate delayed result (valid from previous cycle's coeff read)
                    if (proc_en_d) begin
                        accum <= accum + {{(ACCUM_W-POWER_W-WEIGHT_W){1'b0}}, product};
                    end

                    if (proc_bin == end_bin_r) begin
                        // Last bin presented, need one more cycle for final weight
                        state <= LATCH;
                    end else begin
                        proc_bin <= proc_bin + 1'b1;
                    end
                end

                LATCH: begin
                    // Final weight arrives this cycle, accumulate
                    accum <= accum + {{(ACCUM_W-POWER_W-WEIGHT_W){1'b0}}, product};

                    // Wait one more cycle then write out
                    // (use a registered version to avoid combinational loop)
                    mel_ol[mel_idx] <= accum + {{(ACCUM_W-POWER_W-WEIGHT_W){1'b0}}, product};

                    if (mel_idx == N_MELS - 1) begin
                        valid_ol_r <= 1'b1;
                        mel_idx    <= '0;
                        state      <= STORE;
                    end else begin
                        mel_idx   <= mel_idx + 1'b1;
                        index_addr <= mel_idx + 1'b1;  // pre-fetch start for next filter
                        state     <= LOAD;
                    end
                end

            endcase
        end
    end

endmodule
