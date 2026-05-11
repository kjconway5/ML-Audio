module spectral_vad #(
    parameter int POWER_W   = 31,
    parameter int N_BINS     = 129,
    parameter int THRESH_W   = 32,  // width of programmable threshold
    parameter int CALIB_FRAMES = 256 // power of two for easy shift adjusting
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

    // DFT / debug
    input  logic                dft_vad_obs_en_i,   // enable observation output
    output logic                vad_frame_drop_ol,  // 1 = frame was dropped this cycle (gated by enable)
    
    // output
    output logic               voice_active_ol,  // high when speech detected this frame
    output logic               vad_done_ol       // pulses when decision is made
);

    localparam int BIN_CNT_W = $clog2(N_BINS+1);
    localparam int CALIB_CNT_W = $clog2(CALIB_FRAMES + 1);
    localparam int CALIB_SHIFT = $clog2(CALIB_FRAMES);  // for mean approximation
    // Energy accumulator needs extra bits to sum across calibration frames
    // without overflow: THRESH_W + CALIB_SHIFT bits
    localparam int CALIB_ACC_W = THRESH_W + CALIB_SHIFT;

    // Accumulator & bin counter
    logic [THRESH_W-1:0]   energy_acc_q;
    logic [BIN_CNT_W-1:0]  bin_cnt_q;
    logic                  active_q;
    logic                  done_q;

    // Mode detection
    logic auto_mode;        // threshold_il == 'F
    logic disabled_mode;    // threshold_il == 0
    // Effective threshold used for decisions
    logic [THRESH_W-1:0] eff_threshold;

    // Calibration state
    logic [CALIB_CNT_W-1:0]  calib_cnt_q;     // frame counter during calibration
    logic [CALIB_ACC_W-1:0]  calib_sum_q;     // running sum of frame energies
    logic [THRESH_W-1:0]     auto_thresh_q;   // computed threshold after calibration
    logic                    calibrated_q;    // high once calibration complete

    assign auto_mode     = (threshold_il == {THRESH_W{1'b1}});
    assign disabled_mode = (threshold_il == '0);
    assign eff_threshold = disabled_mode ? '0 :
                            auto_mode    ? auto_thresh_q :
                                        threshold_il;

    // DFT observation: frame was evaluated and dropped
    logic frame_drop_q;

    // Auto-calibration intermediate signals (used inside always_ff)
    logic [CALIB_ACC_W-1:0] final_sum;
    logic [THRESH_W-1:0]    calib_mean;
    logic [THRESH_W-1:0]    margin;
    logic [THRESH_W-1:0]    thresh_candidate;

    // Compute this frame's total energy (with saturation)
    logic [THRESH_W-1:0] frame_energy;
        
    always_ff @(posedge clk_i) begin
        if (reset_i) begin
            energy_acc_q  <= '0;
            bin_cnt_q     <= '0;
            active_q      <= 1'b0;
            done_q        <= 1'b0;
            calib_cnt_q   <= '0;
            calib_sum_q   <= '0;
            auto_thresh_q <= '0;
            calibrated_q  <= 1'b0;
            frame_drop_q  <= 1'b0;
        end else begin
            done_q       <= 1'b0;
            frame_drop_q <= 1'b0;
 
            if (fft_sync_il) begin
                energy_acc_q <= '0;
                bin_cnt_q    <= '0;
            end else if (power_valid_il) begin
                // Saturating accumulator
                if (energy_acc_q + power_il < energy_acc_q)
                    energy_acc_q <= {THRESH_W{1'b1}};
                else
                    energy_acc_q <= energy_acc_q + power_il;
 
                bin_cnt_q <= bin_cnt_q + 1'b1;
 
                if (bin_cnt_q == N_BINS - 1) begin
                    frame_energy = (energy_acc_q + power_il < energy_acc_q)
                                   ? {THRESH_W{1'b1}}
                                   : energy_acc_q + power_il;
 
                    if (auto_mode && !calibrated_q) begin
                        // Calibration period: pass all frames, accumulate sum for mean
                        active_q <= 1'b1;
                        calib_sum_q <= calib_sum_q + {{CALIB_SHIFT{1'b0}}, frame_energy};
                        calib_cnt_q <= calib_cnt_q + 1'b1;
 
                        if (calib_cnt_q == CALIB_FRAMES - 1) begin
                            calibrated_q <= 1'b1;
 
                            final_sum       = calib_sum_q + {{CALIB_SHIFT{1'b0}}, frame_energy};
                            calib_mean      = final_sum[CALIB_ACC_W-1 -: THRESH_W]; // >> CALIB_SHIFT
                            margin          = calib_mean; // change for sclaing factor
                            thresh_candidate = calib_mean + margin; // change for scaling factor
 
                            // Saturate on overflow
                            auto_thresh_q <= (thresh_candidate < calib_mean)
                                             ? {THRESH_W{1'b1}} 
                                             : thresh_candidate;
                        end
                    end else if (disabled_mode) begin
                        active_q <= 1'b1;
                    end else begin
                        active_q     <= frame_energy > eff_threshold;
                        frame_drop_q <= !(frame_energy > eff_threshold);
                    end
 
                    done_q <= 1'b1;
                end
            end
        end
    end
 
    assign voice_active_ol  = active_q;
    assign vad_done_ol      = done_q;
 
    // DFT observation output: pulses high for one cycle when a frame is
    // dropped, but only when the observation enable is asserted.
    // Route to a pad via chip_core for bringup probing.
    assign vad_frame_drop_ol = dft_vad_obs_en_i & frame_drop_q;

endmodule