module log_lut #(
    parameter int ACCUM_W   = 54,
    parameter int N_MELS    = 40,
    parameter int LOG_OUT_W = 16,
    parameter int LUT_FRAC  = 6,
    parameter int Q_FRAC    = 10 // changing to 6.10 because saturating very easily
)(
    input  logic                              clk_i,
    input  logic                              reset_i,

    // from mel_filterbank
    input  logic [N_MELS-1:0][ACCUM_W-1:0]  mel_energy_i,

    // from frame_controller
    input  logic [$clog2(N_MELS)-1:0]        mel_idx_i,
    input  logic                              log_en_i,

    // to output_buffer
    output logic [N_MELS-1:0][LOG_OUT_W-1:0] log_out_o,
    output logic                             log_done_o,

    //Flash write
    input  logic [0:0] flash_write_enable_i,
    input  logic [LUT_FRAC-1:0] flash_addr_i,    
    input  logic [LOG_OUT_W-1:0] flash_write_data_i  
);

    // LUT SRAM
    logic [LUT_FRAC-1:0]  lut_addr;
    logic [LOG_OUT_W-1:0] lut_val;

    log_lut_sram #(
        .LUT_DEPTH (1 << LUT_FRAC),   // 64
        .DATA_W    (LOG_OUT_W),        // 16
        .ADDR_W    (LUT_FRAC)          // 6
    ) u_lut_sram (
        .clk_i       (clk_i),
        .flash_write_enable_i (flash_write_enable_i),
        .flash_addr_i (flash_addr_i),
        .flash_write_data_i (flash_write_data_i),
        .rd_addr_i   (lut_addr),
        .rd_data_o   (lut_val)           // valid 1 cycle after lut_addr
    );

    // Fractional part with LUT
    // extract LUT_FRAC bits just below the leading 1 bit
    // these index into the LUT to get log2(1 + frac)

    assign lut_addr = (log2_int >= LUT_FRAC)
                    ? current_energy >> (log2_int - LUT_FRAC)
                    : current_energy << (LUT_FRAC - log2_int);
    

    // select curr mel energy
    logic [ACCUM_W-1:0] current_energy;
    assign current_energy = mel_energy_i[mel_idx_i];

    // PULP IP to compute floor(log2(cur_energy))
    logic [$clog2(ACCUM_W)-1:0] log2_int;

`ifndef SYNTHESIS
    // Behavioral floor(log2) for simulation (Log2 IP uses constructs unsupported by Icarus)
    always_comb begin
        log2_int = '0;
        for (int k = 0; k < ACCUM_W; k++) begin
            if (current_energy[k])
                log2_int = k;  // implicit truncation; last set bit wins = floor(log2)
        end
    end
`else
    Log2 #(
        .width(ACCUM_W),
        .speed(2)
    ) u_log2 (
        .A(current_energy),
        .Z(log2_int)
    );
`endif


    //Add one cycle delay to log2_int, mel_idx & log_en to align w/ log_result(SRAM delay) 
    logic [ACCUM_W-1:0]           current_energy_d;
    logic [$clog2(ACCUM_W)-1:0]   log2_int_d;
    logic [$clog2(N_MELS)-1:0]    mel_idx_d;
    logic                         log_en_d;
 
    always_ff @(posedge clk_i) begin
        if (reset_i) begin
            current_energy_d <= '0;
            log2_int_d       <= '0;
            mel_idx_d        <= '0;
            log_en_d         <= 1'b0;
        end else begin
            current_energy_d <= current_energy;
            log2_int_d       <= log2_int;
            mel_idx_d        <= mel_idx_i;
            log_en_d         <= log_en_i;
        end
    end


    // Combine integer and fractional parts
    // integer part in Q4.12: floor(log2(x)) shifted left by Q_FRAC
    // fractional part: lut_val already in Q4.12
    // full result = integer + fractional
    // Saturate at max when log2_int exceeds the Q4.12 integer range (15)

    localparam int MAX_LOG_INT = (1 << (LOG_OUT_W - Q_FRAC)) - 1;  // 15
    logic [LOG_OUT_W-1:0] log_result;

    assign log_result = (current_energy_d == '0)
                      ? '0
                      : (log2_int_d > MAX_LOG_INT)
                        ? {LOG_OUT_W{1'b1}}
                        : (log2_int_d << Q_FRAC) + lut_val;

    // Register Output
    // each cycle log_en is high, write result for current mel_idx
    // takes 40 cycles total to compress all mel bins
    // fires log_done_o on the cycle the last bin is written

    always_ff @(posedge clk_i) begin
        if (reset_i) begin
            log_done_o <= 1'b0;
            for (int i = 0; i < N_MELS; i++)
                log_out_o[i] <= '0;
        end else begin
            log_done_o <= 1'b0;

            if (log_en_d) begin
                log_out_o[mel_idx_d] <= log_result;

                if (mel_idx_d == N_MELS-1)
                    log_done_o <= 1'b1;
            end
        end
    end

endmodule
