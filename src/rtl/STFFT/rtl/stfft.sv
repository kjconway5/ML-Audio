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


    // Register reset (matches creator's template pattern)
    reg rst;
    always @(posedge i_clk)
        rst <= i_reset;

    // Windowing
    wire [IW-1:0] win_sample;
    wire          win_ce;


    wire primary_ce = i_ce;

    reg [2:0] alt_delay;
    always @(posedge i_clk) begin
        if (i_reset)
            alt_delay <= 3'b0;
        else
            alt_delay <= {alt_delay[1:0], i_ce};
    end
    wire alt_ce = alt_delay[2];

    windowfn #(
        .IW(IW), .OW(IW), .TW(IW),
        .LGNFFT(FFT_N),
        .OPT_FIXED_TAPS(1'b1),
        .INITIAL_COEFFS("hanning.hex")
    ) u_win (
        .i_clk    (i_clk),
        .i_reset  (i_reset),
        .i_tap_wr (1'b0),
        .i_tap    ({IW{1'b0}}),
        .i_ce     (primary_ce),
        .i_alt_ce (alt_ce),
        .i_sample (i_sample),
        .o_sample (win_sample),
        .o_ce     (win_ce),
        .o_frame  ()
    );
    assign win_ce_o = win_ce;

    // Registered inputs to R2FFT (creator's template pattern)
    reg              sact_istream;
    reg signed [15:0] sdw_istream_real;
    reg signed [15:0] sdw_istream_imag;

    always @(posedge i_clk) begin
        sact_istream     <= win_ce;
        sdw_istream_real <= {{(16-IW){win_sample[IW-1]}}, win_sample};
        sdw_istream_imag <= 16'd0;
    end

    // R2FFT status outputs (registered)
    wire        done_w;
    wire [2:0]  status_w;
    wire signed [7:0] bfpexp_w;

    reg         done_r;
    reg [2:0]   status_r;
    reg signed [7:0] bfpexp_r;

    always @(posedge i_clk) begin
        done_r   <= done_w;
        status_r <= status_w;
        bfpexp_r <= bfpexp_w;
    end
    assign o_bfpexp = bfpexp_r;


    // DMA readout FSM
    reg [FFT_N-1:0] dma_addr;
    reg             dmaact;
    reg             dmaact_r;     // registered version sent to R2FFT
    reg [FFT_N-1:0] dmaa_r;
    reg             readout_done;
    reg             fin_r;

    wire signed [15:0] dmadr_real_w;
    wire signed [15:0] dmadr_imag_w;
    reg  signed [15:0] dmadr_real_r;
    reg  signed [15:0] dmadr_imag_r;

    // Register DMA outputs (creator's pattern)
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
                // Output is registered one extra cycle due to creator's pipeline
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


    // Twiddle ROM/DFFs — registered outputs matching creator's template
    wire [FFT_N-1-2:0] twa_w;
    wire               twact_w;
    reg  [FFT_N-1-2:0] twa_r;
    reg                twact_r;
    wire  [15:0]        twdr_cos_w;


    always @(posedge i_clk) begin
        twact_r   <= twact_w;
        twa_r     <= twa_w;
    end

    fft_twiddle_rom u_twiddle (
        .clk     (i_clk),
        .twact   (twact_r),
        .twa     (twa_r),
        .twdr_cos(twdr_cos_w)
    );


    // Data RAMs — separate rd/wr address ports matching creator's dpram

    // RAM0
    wire               ract_ram0_w, wact_ram0_w;
    wire [FFT_N-2:0]   ra_ram0_w,  wa_ram0_w;
    wire [31:0]        wdw_ram0_w;
    wire [31:0]        rdr_ram0_w;

    reg                ract_ram0_r, wact_ram0_r;
    reg  [FFT_N-2:0]   ra_ram0_r,  wa_ram0_r;
    reg  [31:0]        wdw_ram0_r;

    always @(posedge i_clk) begin
        ract_ram0_r <= ract_ram0_w;
        wact_ram0_r <= wact_ram0_w;
        ra_ram0_r   <= ra_ram0_w;
        wa_ram0_r   <= wa_ram0_w;
        wdw_ram0_r  <= wdw_ram0_w;
    end

    fft_data_ram u_ram0 (
        .clk   (i_clk),
        .ract  (ract_ram0_r),
        .wact  (wact_ram0_r),
        .addr  (wact_ram0_r ? wa_ram0_r : ra_ram0_r),
        .wdata (wdw_ram0_r),
        .rdata (rdr_ram0_w)
    );

    // RAM1
    wire               ract_ram1_w, wact_ram1_w;
    wire [FFT_N-2:0]   ra_ram1_w,  wa_ram1_w;
    wire [31:0]        wdw_ram1_w;
    wire [31:0]        rdr_ram1_w;

    reg                ract_ram1_r, wact_ram1_r;
    reg  [FFT_N-2:0]   ra_ram1_r,  wa_ram1_r;
    reg  [31:0]        wdw_ram1_r;

    always @(posedge i_clk) begin
        ract_ram1_r <= ract_ram1_w;
        wact_ram1_r <= wact_ram1_w;
        ra_ram1_r   <= ra_ram1_w;
        wa_ram1_r   <= wa_ram1_w;
        wdw_ram1_r  <= wdw_ram1_w;
    end

    fft_data_ram u_ram1 (
        .clk   (i_clk),
        .ract  (ract_ram1_r),
        .wact  (wact_ram1_r),
        .addr  (wact_ram1_r ? wa_ram1_r : ra_ram1_r),
        .wdata (wdw_ram1_r),
        .rdata (rdr_ram1_w)
    );


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
        .twdr_cos         (twdr_cos_r),
        .ract_ram0        (ract_ram0_w), .ra_ram0(ra_ram0_w), .rdr_ram0(rdr_ram0_w),
        .wact_ram0        (wact_ram0_w), .wa_ram0(wa_ram0_w), .wdw_ram0(wdw_ram0_w),
        .ract_ram1        (ract_ram1_w), .ra_ram1(ra_ram1_w), .rdr_ram1(rdr_ram1_w),
        .wact_ram1        (wact_ram1_w), .wa_ram1(wa_ram1_w), .wdw_ram1(wdw_ram1_w)
    );

endmodule