module stfft #(
    parameter IW       = 16,
    parameter OW       = 16,
    parameter FFT_SIZE = 256,
    parameter FFT_N    = $clog2(FFT_SIZE)
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

    // ----------------------------------------------------------------
    // Registered reset (creator's template)
    // ----------------------------------------------------------------
    reg rst;
    always @(posedge i_clk)
        rst <= i_reset;

    // ----------------------------------------------------------------
    // Simple Hanning window — 3-cycle pipeline matching windowfn depth
    // ----------------------------------------------------------------
    reg [IW-1:0] hanning_rom [0:FFT_SIZE-1];
    initial $readmemh("hanning.hex", hanning_rom);

    // Stage 1: register sample and fetch coefficient
    reg signed [IW-1:0]   s1_sample;
    reg        [IW-1:0]   s1_coeff;
    reg                   s1_ce;
    reg [FFT_N-1:0]       sample_cnt;

    always @(posedge i_clk) begin
        if (i_reset)
            sample_cnt <= 0;
        else if (i_ce)
            sample_cnt <= sample_cnt + 1;
    end

    always @(posedge i_clk) begin
        s1_sample <= i_sample;
        s1_coeff  <= hanning_rom[sample_cnt];
        s1_ce     <= i_ce;
    end

    // Stage 2: multiply (signed sample * unsigned Hanning coefficient)
    reg signed [2*IW-1:0] s2_product;
    reg                   s2_ce;

    always @(posedge i_clk) begin
        s2_product <= s1_sample * $signed({1'b0, s1_coeff}); // Q1.15
        s2_ce      <= s1_ce;
    end

    // Stage 3: round and truncate to IW bits
    wire signed [IW-1:0] win_sample_w = s2_product[2*IW-2 : IW-1];
    reg  signed [IW-1:0] win_sample;
    reg                  win_ce;

    always @(posedge i_clk) begin
        win_sample <= win_sample_w;
        win_ce     <= s2_ce;
    end

    assign win_ce_o = win_ce;

    // ----------------------------------------------------------------
    // Registered inputs to R2FFT
    // ----------------------------------------------------------------
    reg              sact_istream;
    reg signed [15:0] sdw_istream_real;
    reg signed [15:0] sdw_istream_imag;

    always @(posedge i_clk) begin
        sact_istream     <= win_ce;
        sdw_istream_real <= {{(16-IW){win_sample[IW-1]}}, win_sample};
        sdw_istream_imag <= 16'd0;
    end

    // ----------------------------------------------------------------
    // R2FFT status (registered)
    // ----------------------------------------------------------------
    wire        done_w;
    wire [2:0]  status_w;
    wire signed [7:0] bfpexp_w;
    reg         done_r;
    reg signed [7:0] bfpexp_r;

    always @(posedge i_clk) begin
        done_r   <= done_w;
        bfpexp_r <= bfpexp_w;
    end
    assign o_bfpexp = bfpexp_r;

    // ----------------------------------------------------------------
    // DMA readout FSM
    // ----------------------------------------------------------------
    reg [FFT_N-1:0]    dma_addr;
    reg                dmaact, dmaact_r;
    reg [FFT_N-1:0]    dmaa_r;
    reg                readout_done, fin_r;

    wire signed [15:0] dmadr_real_w, dmadr_imag_w;
    reg  signed [15:0] dmadr_real_r, dmadr_imag_r;

    always @(posedge i_clk) begin
        dmaact_r    <= dmaact;
        dmaa_r      <= dma_addr;
        dmadr_real_r <= dmadr_real_w;
        dmadr_imag_r <= dmadr_imag_w;
        fin_r        <= readout_done;
    end

    always @(posedge i_clk) begin
        if (i_reset) begin
            dmaact       <= 1'b0;
            dma_addr     <= '0;
            readout_done <= 1'b0;
            o_fft_sync   <= 1'b0;
            o_fft_result <= '0;
        end else begin
            o_fft_sync <= 1'b0;

            if (done_r && !readout_done && !dmaact) begin
                dmaact     <= 1'b1;
                dma_addr   <= '0;
                o_fft_sync <= 1'b1;
            end else if (dmaact) begin
                o_fft_result <= {
                    {{(OW-16){dmadr_real_r[15]}}, dmadr_real_r},
                    {{(OW-16){dmadr_imag_r[15]}}, dmadr_imag_r}
                };
                dma_addr <= dma_addr + 1;
                if (dma_addr == FFT_SIZE-1) begin
                    dmaact       <= 1'b0;
                    readout_done <= 1'b1;
                end
            end else if (readout_done) begin
                readout_done <= 1'b0;
            end
        end
    end

    // ----------------------------------------------------------------
    // Twiddle ROM (direct connection — fft_twiddle_rom already registered)
    // ----------------------------------------------------------------
    wire [FFT_N-1-2:0] twa_w;
    wire               twact_w;
    wire [15:0]        twdr_cos_w;

    fft_twiddle_rom u_twiddle (
        .clk     (i_clk),
        .twact   (twact_w),
        .twa     (twa_w),
        .twdr_cos(twdr_cos_w)
    );

    // ----------------------------------------------------------------
    // Data RAMs (direct connection — fft_data_ram already registered)
    // ----------------------------------------------------------------
    wire               ract_ram0_w, wact_ram0_w;
    wire [FFT_N-2:0]   ra_ram0_w,  wa_ram0_w;
    wire [31:0]        wdw_ram0_w, rdr_ram0_w;

    fft_data_ram u_ram0 (
        .clk   (i_clk),
        .ract  (ract_ram0_w),
        .ra    (ra_ram0_w),
        .rdata (rdr_ram0_w),
        .wact  (wact_ram0_w),
        .wa    (wa_ram0_w),
        .wdata (wdw_ram0_w)
    );

    wire               ract_ram1_w, wact_ram1_w;
    wire [FFT_N-2:0]   ra_ram1_w,  wa_ram1_w;
    wire [31:0]        wdw_ram1_w, rdr_ram1_w;

    fft_data_ram u_ram1 (
        .clk   (i_clk),
        .ract  (ract_ram1_w),
        .ra    (ra_ram1_w),
        .rdata (rdr_ram1_w),
        .wact  (wact_ram1_w),
        .wa    (wa_ram1_w),
        .wdata (wdw_ram1_w)
    );

    // ----------------------------------------------------------------
    // R2FFT core
    // ----------------------------------------------------------------
    R2FFT #(
        .FFT_LENGTH(FFT_SIZE),
        .FFT_DW(16),
        .PL_DEPTH(3)
    ) u_r2fft (
        .clk              (i_clk),
        .rst              (rst),
        .autorun          (1'b1),
        .run              (1'b0),
        .fin              (fin_r),
        .ifft             (1'b0),
        .done             (done_w),
        .status           (status_w),
        .bfpexp           (bfpexp_w),
        .sact_istream     (sact_istream),
        .sdw_istream_real (sdw_istream_real),
        .sdw_istream_imag (sdw_istream_imag),
        .dmaact           (dmaact_r),
        .dmaa             (dmaa_r),
        .dmadr_real       (dmadr_real_w),
        .dmadr_imag       (dmadr_imag_w),
        .twact            (twact_w),
        .twa              (twa_w),
        .twdr_cos         (twdr_cos_w),
        .ract_ram0        (ract_ram0_w), .ra_ram0(ra_ram0_w), .rdr_ram0(rdr_ram0_w),
        .wact_ram0        (wact_ram0_w), .wa_ram0(wa_ram0_w), .wdw_ram0(wdw_ram0_w),
        .ract_ram1        (ract_ram1_w), .ra_ram1(ra_ram1_w), .rdr_ram1(rdr_ram1_w),
        .wact_ram1        (wact_ram1_w), .wa_ram1(wa_ram1_w), .wdw_ram1(wdw_ram1_w)
    );

endmodule