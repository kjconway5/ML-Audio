module mel_filterbank #(
    parameter int N_MELS     = 40,
    parameter int N_BINS     = 129,
    parameter int MAX_COEFFS = 16,
    parameter int POWER_W    = 31,
    parameter int WEIGHT_W   = 16,
    parameter int ACCUM_W    = 54
)(
    input  logic [0:0] clk_i,
    input  logic [0:0]  reset_i,
    input  logic [POWER_W-1:0] power_il,
    input  logic [0:0] valid_il,
    output logic [N_MELS-1:0][ACCUM_W-1:0] mel_ol,
    output logic [0:0] valid_ol
);

    //Power sample buffer
    logic [POWER_W-1:0] power_buf [N_BINS];
    logic [7:0]         store_ctr;   // 0..N_BINS-1

    // Per-filter processing state
    logic [$clog2(N_MELS)-1:0] mel_idx;
    logic [7:0]                proc_bin;    // current FFT bin being MAC'd
    logic [7:0]                start_bin_r, end_bin_r; // latched from ROM
    logic [ACCUM_W-1:0]        accum;

    // Combinational read
    logic [$clog2(MAX_COEFFS)-1:0] coeff_idx;
    logic [WEIGHT_W-1:0]           weight;
    logic [7:0]                    rom_start, rom_end;

    //offset of proc_bin 
    assign coeff_idx = proc_bin - start_bin_r;

    mel_coeff_rom #(
        .MEL_BINS   (N_MELS),
        .MAX_COEFFS (MAX_COEFFS),
        .COEFF_W    (WEIGHT_W),
        .BIN_W      (8)
    ) u_rom (
        .mel_idx   (mel_idx),
        .coeff_idx (coeff_idx),
        .weight_out(weight),
        .start_bin (rom_start),
        .end_bin   (rom_end)
    );

    // MAC product 
    logic [POWER_W+WEIGHT_W-1:0] product;

//Help Simulation run faster 
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

    // FSM
    typedef enum logic [1:0] { STORE, LOAD, PROC, LATCH } state_t;
    state_t state;

    logic valid_ol_r;
    assign valid_ol = valid_ol_r;

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
            for (int i = 0; i < N_MELS; i++) mel_ol[i] <= '0;
        end else begin
            // default: deassert every cycle
            valid_ol_r <= 1'b0;   

            case (state)
                // STORE: buffer incoming power bins one per valid_il pulse
                  STORE: begin
                    if (valid_il) begin
                        power_buf[store_ctr] <= power_il;
                        if (store_ctr == N_BINS - 1) begin
                            store_ctr <= '0;
                            mel_idx   <= '0;
                            state     <= LOAD;
                        end else begin
                            store_ctr <= store_ctr + 1'b1;
                        end
                    end
                end

                // LOAD: latch start/end bins from ROM for current mel_idx
                LOAD: begin
                    start_bin_r <= rom_start;
                    end_bin_r   <= rom_end;
                    proc_bin    <= rom_start;
                    accum       <= '0;
                    state       <= PROC;
                end
                // PROC: accumulate power_buf[proc_bin] * weight each cycle
                PROC: begin
                    accum <= accum + {{(ACCUM_W-POWER_W-WEIGHT_W){1'b0}}, product};
                    if (proc_bin == end_bin_r)
                        state <= LATCH;
                    else
                        proc_bin <= proc_bin + 1'b1;
                end

                // LATCH: capture accumulator into mel_ol, advance to next mel
                LATCH: begin
                    mel_ol[mel_idx] <= accum;
                    if (mel_idx == N_MELS - 1) begin
                        // All filters done: pulse valid_ol, restart for next frame
                        valid_ol_r <= 1'b1;
                        mel_idx    <= '0;
                        state      <= STORE;
                    end else begin
                        mel_idx <= mel_idx + 1'b1;
                        state   <= LOAD;
                    end
                end
            endcase
        end
    end

endmodule
