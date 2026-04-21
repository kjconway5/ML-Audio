// stfft.sv
//
// Short-Time FFT with 50% overlap implemented via two ping-pong R2FFT instances.
//
// Architecture:
//   - HOP = FFT_SIZE/2 samples between consecutive output frames.
//   - Instance A processes windows starting at samples 0, 256, 512, ...
//   - Instance B processes windows starting at samples 128, 384, 640, ...
//   - Both share the same Hanning ROM (read-only; accesses don't conflict for
//     CE_EVERY >= 2 because A and B compute at non-overlapping times).
//
// Timing requirement (must be met by the caller):
//   CE_EVERY * HOP > logmel_processing_time
//   With HOP=128 and logmel taking ~7500 clocks: CE_EVERY >= 59  (use 64+).
//
// DMA non-overlap guarantee:
//   A's DMA ends at ~(FFT_SIZE * CE_EVERY + FFT_COMPUTE + FFT_SIZE) clocks.
//   B's DMA starts at ~((FFT_SIZE+HOP) * CE_EVERY + FFT_COMPUTE) clocks.
//   These don't overlap when CE_EVERY >= 2 (always satisfied in practice).
//
// Pipeline_top interface is UNCHANGED:
//   o_fft_sync fires 1 cycle before the first valid o_fft_result word.
//   o_fft_result streams FFT_SIZE bins at full clock speed after each sync.
//   o_bfpexp is stable for the duration of each DMA readout.

module stfft #(
    parameter IW       = 16,
    parameter OW       = 16,
    parameter FFT_SIZE = 256,
    parameter FFT_N    = $clog2(FFT_SIZE),
    parameter HOP      = FFT_SIZE / 2   // 50% overlap; must divide FFT_SIZE evenly
)(
    input  wire             i_clk,
    input  wire             i_reset,
    input  wire             i_ce,
    input  wire [IW-1:0]    i_sample,
    output reg  [2*OW-1:0]  o_fft_result,
    output reg              o_fft_sync,
    output wire             win_ce_o,
    output wire signed [7:0] o_bfpexp
);

    // =========================================================================
    // Delayed reset for R2FFT (matches original)
    // =========================================================================
    reg rst;
    always @(posedge i_clk) rst <= i_reset;

    // Hanning window ROM — shared by both channels (read-only, no conflict)
    reg [IW-1:0] hanning_rom [0:FFT_SIZE-1];
    initial $readmemh("hanning.hex", hanning_rom);

    // =========================================================================
    // HOP boundary tracker and A/B channel selector
    //
    // hop_stb: pulses once per HOP input samples (at the first sample of each
    //          HOP group).  ab_sel alternates at every hop_stb.
    //   ab_sel=0 → A's window starts at this HOP boundary
    //   ab_sel=1 → B's window starts at this HOP boundary
    // =========================================================================
    localparam HOP_BITS = $clog2(HOP);

    reg [HOP_BITS-1:0] hop_pos;
    reg                ab_sel;

    wire hop_stb = i_ce && (hop_pos == {HOP_BITS{1'b0}});

    always @(posedge i_clk) begin
        if (i_reset) begin
            hop_pos <= {HOP_BITS{1'b0}};
            ab_sel  <= 1'b0;
        end else if (i_ce) begin
            hop_pos <= (hop_pos == HOP - 1) ? {HOP_BITS{1'b0}} : hop_pos + 1'b1;
            if (hop_stb)
                ab_sel <= ~ab_sel;
        end
    end

    // =========================================================================
    // CHANNEL A — windowing pipeline (3 stages)
    //
    // win_cnt_a: position within A's current window.
    //   FFT_SIZE = inactive sentinel (no in-progress frame)
    //   0..FFT_SIZE-1 = active, tracking which Hanning coefficient to apply
    //
    // a_starting: the very first sample of A's new window is being clocked in
    //             this cycle.  Position 0 gets hanning[0] immediately.
    // =========================================================================
    reg [FFT_N:0] win_cnt_a;   // one extra bit so FFT_SIZE fits as inactive value

    wire a_starting = hop_stb && !ab_sel;
    wire a_active   = (win_cnt_a < FFT_SIZE);
    wire a_win_en   = a_starting || (i_ce && a_active);

    always @(posedge i_clk) begin
        if (i_reset)
            win_cnt_a <= FFT_SIZE;   // inactive
        else if (a_starting)
            win_cnt_a <= 1;          // position 0 consumed this cycle; next is 1
        else if (i_ce && a_active)
            win_cnt_a <= win_cnt_a + 1'b1;
    end

    // Hanning index: position 0 on start, otherwise the running counter
    wire [FFT_N-1:0] a_coeff_idx = a_starting ? {FFT_N{1'b0}} : win_cnt_a[FFT_N-1:0];

    // Stage 1: latch sample + coefficient
    reg signed [IW-1:0]   a_s1_samp;
    reg        [IW-1:0]   a_s1_coeff;
    reg                   a_s1_ce;

    always @(posedge i_clk) begin
        a_s1_samp  <= $signed(i_sample);
        a_s1_coeff <= hanning_rom[a_coeff_idx];
        a_s1_ce    <= a_win_en;
    end

    // Stage 2: multiply (signed sample × unsigned Hanning, Q1.15 result)
    reg signed [2*IW-1:0] a_s2_prod;
    reg                   a_s2_ce;

    always @(posedge i_clk) begin
        a_s2_prod <= a_s1_samp * $signed({1'b0, a_s1_coeff});
        a_s2_ce   <= a_s1_ce;
    end

    // Stage 3: round-and-truncate to IW bits
    wire signed [IW-1:0] a_win_samp_w = a_s2_prod[2*IW-2 : IW-1];
    reg  signed [IW-1:0] a_win_samp;
    reg                  a_win_ce;

    always @(posedge i_clk) begin
        a_win_samp <= a_win_samp_w;
        a_win_ce   <= a_s2_ce;
    end

    // =========================================================================
    // CHANNEL B — same 3-stage windowing, triggered at odd HOP boundaries
    // =========================================================================
    reg [FFT_N:0] win_cnt_b;

    wire b_starting = hop_stb && ab_sel;
    wire b_active   = (win_cnt_b < FFT_SIZE);
    wire b_win_en   = b_starting || (i_ce && b_active);

    always @(posedge i_clk) begin
        if (i_reset)
            win_cnt_b <= FFT_SIZE;
        else if (b_starting)
            win_cnt_b <= 1;
        else if (i_ce && b_active)
            win_cnt_b <= win_cnt_b + 1'b1;
    end

    wire [FFT_N-1:0] b_coeff_idx = b_starting ? {FFT_N{1'b0}} : win_cnt_b[FFT_N-1:0];

    reg signed [IW-1:0]   b_s1_samp;
    reg        [IW-1:0]   b_s1_coeff;
    reg                   b_s1_ce;

    always @(posedge i_clk) begin
        b_s1_samp  <= $signed(i_sample);
        b_s1_coeff <= hanning_rom[b_coeff_idx];
        b_s1_ce    <= b_win_en;
    end

    reg signed [2*IW-1:0] b_s2_prod;
    reg                   b_s2_ce;

    always @(posedge i_clk) begin
        b_s2_prod <= b_s1_samp * $signed({1'b0, b_s1_coeff});
        b_s2_ce   <= b_s1_ce;
    end

    wire signed [IW-1:0] b_win_samp_w = b_s2_prod[2*IW-2 : IW-1];
    reg  signed [IW-1:0] b_win_samp;
    reg                  b_win_ce;

    always @(posedge i_clk) begin
        b_win_samp <= b_win_samp_w;
        b_win_ce   <= b_s2_ce;
    end

    // win_ce_o: high whenever either channel is outputting a windowed sample
    assign win_ce_o = a_win_ce | b_win_ce;


    // R2FFT INSTANCE A

    fft fft_core1
    #(
        .FFT_LENGTH(),
        .FFT_DW(),
        .PL_DEPTH(),
        .FFT_N()
    )
    (
        // system
        .clk,
        .rst_i,

        // control
        .autorun_i,
        .run_i,
        .fin_i,
        .ifft_i,

        // status
        .done_o,
        .status_o,
        .bfpexp_o,

        // input stream
        .sact_istream_i,
        .sdw_istream_real_i,
        .sdw_istream_imag_i,

        // DMA readout
        .dmaact_i,
        .dmaa_i,
        .dmadr_real_o,
        .dmadr_imag_o
    );

    // R2FFT INSTANCE B  (identical structure, fed by Channel B windowing)

    fft fft_core1
    #(
        .FFT_LENGTH(),
        .FFT_DW(),
        .PL_DEPTH(),
        .FFT_N()
    )
    (
        // system
        .clk,
        .rst_i,

        // control
        .autorun_i,
        .run_i,
        .fin_i,
        .ifft_i,

        // status
        .done_o,
        .status_o,
        .bfpexp_o,

        // input stream
        .sact_istream_i,
        .sdw_istream_real_i,
        .sdw_istream_imag_i,

        // DMA readout
        .dmaact_i,
        .dmaa_i,
        .dmadr_real_o,
        .dmadr_imag_o
    );

    // =========================================================================
    // Output MUX
    //
    // A and B DMA readouts never overlap for CE_EVERY >= 2.
    // o_fft_sync fires 1 cycle before the first valid o_fft_result word,
    // preserving the same timing contract as the single-instance version.
    // o_bfpexp reflects whichever instance is currently streaming output.
    // =========================================================================
    always @(posedge i_clk) begin
        if (i_reset) begin
            o_fft_sync   <= 1'b0;
            o_fft_result <= {2*OW{1'b0}};
        end else begin
            o_fft_sync <= a_sync | b_sync;
            if (a_dmaact)
                o_fft_result <= a_result;
            else if (b_dmaact)
                o_fft_result <= b_result;
            // else: hold (unimportant; pipeline_top gates on fft_valid)
        end
    end

    assign o_bfpexp = a_dmaact ? a_bfpexp_r : b_bfpexp_r;

endmodule