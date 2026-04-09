module stfft #(
    parameter IW       = 14,
    parameter OW       = 18,
    parameter FFT_SIZE = 256
)(
    input  wire             i_clk,
    input  wire             i_reset,
    input  wire             i_ce,
    input  wire [IW-1:0]    i_sample,
    output reg  [2*OW-1:0]  o_fft_result,
    output reg              o_fft_sync,
    output wire             win_ce_o,
    // BFP exponent output — connect to filterbank
    output wire signed [7:0] o_bfpexp
);
    localparam FFT_N = $clog2(FFT_SIZE);  // 8 for 256-point

    // ----------------------------------------------------------------
    // Windowing — unchanged from your existing code
    // ----------------------------------------------------------------
    wire [IW-1:0] win_sample;
    wire          win_ce;

    reg [2:0] alt_delay;
    always @(posedge i_clk)
        alt_delay <= i_reset ? 3'b0 : {alt_delay[1:0], i_ce};

    windowfn #(
        .IW(IW), .OW(IW), .TW(IW),
        .LGNFFT($clog2(FFT_SIZE)),
        .INITIAL_COEFFS("hanning.hex")
    ) u_win (
        .i_clk(i_clk), .i_reset(i_reset),
        .i_tap_wr(1'b0), .i_tap({IW{1'b0}}),
        .i_ce(i_ce), .i_alt_ce(alt_delay[2]),
        .i_sample(i_sample),
        .o_sample(win_sample),
        .o_ce(win_ce),
        .o_frame()
    );
    assign win_ce_o = win_ce;

    // ----------------------------------------------------------------
    // R2FFT wires
    // ----------------------------------------------------------------
    wire        twact;
    wire [5:0]  twa;       // FFT_N-1-2 = 5 bits
    wire [15:0] twdr_cos;

    wire        ract_ram0, wact_ram0;
    wire [6:0]  ra_ram0,  wa_ram0;   // FFT_N-1-1 = 6 bits → 7-bit addr
    wire [31:0] rdr_ram0, wdw_ram0;

    wire        ract_ram1, wact_ram1;
    wire [6:0]  ra_ram1,  wa_ram1;
    wire [31:0] rdr_ram1, wdw_ram1;

    wire        r2fft_done;
    wire [2:0]  r2fft_status;

    // DMA readout controller signals
    reg  [FFT_N-1:0] dma_addr;
    reg              dmaact;
    reg              readout_done;
    wire [15:0]      dmadr_real, dmadr_imag;
    reg              dmaa_lsb_d;  // registered lsb for output mux

    // ----------------------------------------------------------------
    // R2FFT core
    // ----------------------------------------------------------------
    R2FFT #(
        .FFT_LENGTH(FFT_SIZE),
        .FFT_DW(16),
        .PL_DEPTH(3)
    ) u_r2fft (
        .clk             (i_clk),
        .rst             (i_reset),
        .autorun         (1'b1),
        .run             (1'b0),
        .fin             (readout_done),
        .ifft            (1'b0),
        .done            (r2fft_done),
        .status          (r2fft_status),
        .bfpexp          (o_bfpexp),

        .sact_istream    (win_ce),
        .sdw_istream_real({{(16-IW){win_sample[IW-1]}}, win_sample}),
        .sdw_istream_imag(16'd0),

        .dmaact          (dmaact),
        .dmaa            (dma_addr),
        .dmadr_real      (dmadr_real),
        .dmadr_imag      (dmadr_imag),

        .twact           (twact),
        .twa             (twa),
        .twdr_cos        (twdr_cos),

        .ract_ram0       (ract_ram0), .ra_ram0(ra_ram0), .rdr_ram0(rdr_ram0),
        .wact_ram0       (wact_ram0), .wa_ram0(wa_ram0), .wdw_ram0(wdw_ram0),
        .ract_ram1       (ract_ram1), .ra_ram1(ra_ram1), .rdr_ram1(rdr_ram1),
        .wact_ram1       (wact_ram1), .wa_ram1(wa_ram1), .wdw_ram1(wdw_ram1)
    );

    // ----------------------------------------------------------------
    // Memory instances
    // ----------------------------------------------------------------
    fft_twiddle_rom u_twiddle (
        .clk(i_clk), .twact(twact), .twa(twa), .twdr_cos(twdr_cos)
    );

    fft_data_ram u_ram0 (
        .clk(i_clk),
        .ract(ract_ram0), .wact(wact_ram0),
        .addr(wact_ram0 ? wa_ram0 : ra_ram0),
        .wdata(wdw_ram0), .rdata(rdr_ram0)
    );

    fft_data_ram u_ram1 (
        .clk(i_clk),
        .ract(ract_ram1), .wact(wact_ram1),
        .addr(wact_ram1 ? wa_ram1 : ra_ram1),
        .wdata(wdw_ram1), .rdata(rdr_ram1)
    );

    // ----------------------------------------------------------------
    // DMA readout FSM
    // ----------------------------------------------------------------
    always @(posedge i_clk) begin
        if (i_reset) begin
            dmaact       <= 1'b0;
            dma_addr     <= '0;
            readout_done <= 1'b0;
            o_fft_sync   <= 1'b0;
        end else begin
            o_fft_sync <= 1'b0;

            if (r2fft_done && !readout_done && !dmaact) begin
                // start readout
                dmaact   <= 1'b1;
                dma_addr <= '0;
                o_fft_sync <= 1'b1;  // sync pulse on first bin
            end else if (dmaact) begin
                // capture output (registered lsb from R2FFT's internal mux)
                // present result to downstream pipeline
                o_fft_result <= {
                    {{(OW-16){dmadr_imag[15]}}, dmadr_imag},
                    {{(OW-16){dmadr_real[15]}}, dmadr_real}
                };
                dma_addr <= dma_addr + 1;
                if (dma_addr == FFT_SIZE-1) begin
                    dmaact       <= 1'b0;
                    readout_done <= 1'b1;
                end
            end else if (readout_done) begin
                // hold until R2FFT acknowledges via fin, then clear
                readout_done <= 1'b0;
            end
        end
    end

endmodule