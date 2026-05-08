module spectral_vad #(
    parameter int POWER_W   = 31,
    parameter int N_BINS     = 129,
    parameter int THRESH_W   = 32  // width of programmable threshold
)(
    input  logic              clk_i,
    input  logic              reset_i,
    
    // from power_calc
    input  logic [POWER_W-1:0] power_il,
    input  logic               power_valid_il,
    
    // from STFFT (frame boundary)
    input  logic               fft_sync_il,
    
    // programmable threshold (loaded via UART boot)
    input  logic [THRESH_W-1:0] threshold_il,
    
    // output
    output logic               voice_active_ol,  // high when speech detected this frame
    output logic               vad_done_ol       // pulses when decision is made
);

    localparam int BIN_CNT_W = $clog2(N_BINS+1);
    
    logic [THRESH_W-1:0]   energy_acc_q;
    logic [BIN_CNT_W-1:0]  bin_cnt_q;
    logic                  active_q;
    logic                  done_q;
    
    always_ff @(posedge clk_i) begin
        if (reset_i) begin
            energy_acc_q <= '0;
            bin_cnt_q    <= '0;
            active_q     <= 1'b0;
            done_q       <= 1'b0;
        end else begin
            done_q <= 1'b0;
            
            if (fft_sync_il) begin
                // new frame starting, reset accumulator
                energy_acc_q <= '0;
                bin_cnt_q    <= '0;
            end else if (power_valid_il) begin
                // accumulate power, saturate on overflow
                if (energy_acc_q + power_il < energy_acc_q)
                    energy_acc_q <= {THRESH_W{1'b1}};  // saturate
                else
                    energy_acc_q <= energy_acc_q + power_il;
                    
                bin_cnt_q <= bin_cnt_q + 1'b1;
                
                // last bin, make decision
                if (bin_cnt_q == N_BINS - 1) begin
                    active_q <= (energy_acc_q + power_il) > threshold_il;
                    done_q   <= 1'b1;
                end
            end
        end
    end
    
    assign voice_active_ol = active_q;
    assign vad_done_ol     = done_q;

endmodule