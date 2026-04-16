module stfft #(
    parameter IW       = 16,
    parameter OW       = 16,
    parameter FFT_SIZE = 256
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

    localparam FFT_N = $clog2(FFT_SIZE);

    wire [IW-1:0] win_sample;
    wire          win_ce;

    reg [2:0] alt_delay;
    always @(posedge i_clk) begin
        if (i_reset)
            alt_delay <= 3'b0;
        else
            alt_delay <= {alt_delay[1:0], i_ce};
    end

    windowfn #(
        .IW(IW), .OW(IW), .TW(IW),
        .LGNFFT(FFT_N),
        .INITIAL_COEFFS("hanning.hex")
    ) u_win (
        .i_clk(i_clk), .i_reset(i_reset),
        .i_tap_wr(1'b0), .i_tap({IW{1'b0}}),
        .i_ce(i_ce),
        .i_alt_ce(alt_delay[2]),
        .i_sample(i_sample),
        .o_sample(win_sample),
        .o_ce(win_ce),
        .o_frame()
    );

    assign win_ce_o = win_ce;


    reg [FFT_N-1:0] sample_cnt;
    reg             running;

    reg             sact_istream;
    reg signed [15:0] sdw_real, sdw_imag;

    reg run, fin;

    always @(posedge i_clk) begin
        if (i_reset) begin
            sample_cnt    <= 0;
            running       <= 0;
            run           <= 0;
            fin           <= 0;
            sact_istream  <= 0;
        end else begin
            run <= 0;
            fin <= 0;
            sact_istream <= 0;

            if (win_ce) begin
                sact_istream <= 1;

                // start frame
                if (!running) begin
                    run <= 1;
                    running <= 1;
                    sample_cnt <= 0;
                end

                // data
                sdw_real <= {{(16-IW){win_sample[IW-1]}}, win_sample};
                sdw_imag <= 16'd0;

                // end frame
                if (sample_cnt == FFT_SIZE-1) begin
                    fin <= 1;
                    running <= 0;
                end

                sample_cnt <= sample_cnt + 1;
            end
        end
    end

    wire        twact;
    wire [FFT_N-3:0] twa;
    wire [15:0] twdr_cos;

    wire        ract_ram0, wact_ram0;
    wire [FFT_N-2:0] ra_ram0, wa_ram0;
    wire [31:0] rdr_ram0, wdw_ram0;

    wire        ract_ram1, wact_ram1;
    wire [FFT_N-2:0] ra_ram1, wa_ram1;
    wire [31:0] rdr_ram1, wdw_ram1;

    wire        r2fft_done;
    wire [2:0]  r2fft_status;


    R2FFT #(
        .FFT_LENGTH(FFT_SIZE),
        .FFT_DW(16),
        .PL_DEPTH(3)
    ) u_r2fft (
        .clk(i_clk),
        .rst(i_reset),

        .autorun(1'b0),   // FIXED
        .run(run),
        .fin(fin),
        .ifft(1'b0),

        .done(r2fft_done),
        .status(r2fft_status),
        .bfpexp(o_bfpexp),

        .sact_istream(sact_istream),
        .sdw_istream_real(sdw_real),
        .sdw_istream_imag(sdw_imag),

        .dmaact(dmaact),
        .dmaa(dma_addr),
        .dmadr_real(dmadr_real),
        .dmadr_imag(dmadr_imag),

        .twact(twact),
        .twa(twa),
        .twdr_cos(twdr_cos),

        .ract_ram0(ract_ram0), .ra_ram0(ra_ram0), .rdr_ram0(rdr_ram0),
        .wact_ram0(wact_ram0), .wa_ram0(wa_ram0), .wdw_ram0(wdw_ram0),

        .ract_ram1(ract_ram1), .ra_ram1(ra_ram1), .rdr_ram1(rdr_ram1),
        .wact_ram1(wact_ram1), .wa_ram1(wa_ram1), .wdw_ram1(wdw_ram1)
    );


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

    reg [FFT_N-1:0] dma_addr;
    reg             dmaact;
    wire [15:0]     dmadr_real, dmadr_imag;

    always @(posedge i_clk) begin
        if (i_reset) begin
            dmaact     <= 0;
            dma_addr   <= 0;
            o_fft_sync <= 0;
        end else begin
            o_fft_sync <= 0;

            if (r2fft_done && !dmaact) begin
                dmaact     <= 1;
                dma_addr   <= 0;
                o_fft_sync <= 1;
            end else if (dmaact) begin
                o_fft_result <= {
                    {{(OW-16){dmadr_imag[15]}}, dmadr_imag},
                    {{(OW-16){dmadr_real[15]}}, dmadr_real}
                };

                dma_addr <= dma_addr + 1;

                if (dma_addr == FFT_SIZE-1)
                    dmaact <= 0;
            end
        end
    end

endmodule