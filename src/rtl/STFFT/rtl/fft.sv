module fft
  #(
    parameter FFT_LENGTH = 1024,
    parameter FFT_DW = 16,
    parameter PL_DEPTH = 3,
    parameter FFT_N = $clog2(FFT_LENGTH)
  )
(
    // system
    input  wire clk,
    input  wire rst_i,

    // control
    input  wire autorun_i,
    input  wire run_i,
    input  wire fin_i,
    input  wire ifft_i,

    // status
    output reg done_o,
    output reg [2:0] status_o,
    output reg signed [7:0] bfpexp_o,

    // input stream
    input  wire sact_istream_i,
    input  wire signed [FFT_DW-1:0] sdw_istream_real_i,
    input  wire signed [FFT_DW-1:0] sdw_istream_imag_i,

    // DMA readout
    input  wire dmaact_i,
    input  wire [FFT_N-1:0] dmaa_i,
    output reg signed [FFT_DW-1:0] dmadr_real_o,
    output reg signed [FFT_DW-1:0] dmadr_imag_o
);

    reg rst;


    reg sact;
    reg signed [FFT_DW-1:0] sreal, simag;

    reg autorun, run, fin, ifft;

    always @(posedge clk) begin
        rst <= rst_i;

        sact   <= sact_istream_i;
        sreal  <= sdw_istream_real_i;
        simag  <= sdw_istream_imag_i;

        autorun <= autorun_i;
        run     <= run_i;
        fin     <= fin_i;
        ifft    <= ifft_i;
    end


    wire done;
    wire [2:0] status;
    wire signed [7:0] bfpexp;

    always @(posedge clk) begin
        done_o   <= done;
        status_o <= status;
        bfpexp_o <= bfpexp;
    end


    reg dmaact;
    reg [FFT_N-1:0] dmaa;

    wire signed [FFT_DW-1:0] dmadr_real;
    wire signed [FFT_DW-1:0] dmadr_imag;

    always @(posedge clk) begin
        dmaact <= dmaact_i;
        dmaa   <= dmaa_i;

        dmadr_real_o <= dmadr_real;
        dmadr_imag_o <= dmadr_imag;
    end


    wire twact;
    wire [FFT_N-3:0] twa;
    wire [FFT_DW-1:0] twdr_cos;

    fft_twiddle_rom u_tw (
        .clk(clk),
        .twact(twact),
        .twa(twa),
        .twdr_cos(twdr_cos)
    );


    wire ract0, wact0;
    wire [FFT_N-2:0] ra0, wa0;
    wire [31:0] w0, r0;

    wire ract1, wact1;
    wire [FFT_N-2:0] ra1, wa1;
    wire [31:0] w1, r1;

    fft_data_ram ram0 (
        .clk(clk),
        .ract(ract0),
        .wact(wact0),
        .addr(wact0 ? wa0 : ra0),
        .wdata(w0),
        .rdata(r0)
    );

    fft_data_ram ram1 (
        .clk(clk),
        .ract(ract1),
        .wact(wact1),
        .addr(wact1 ? wa1 : ra1),
        .wdata(w1),
        .rdata(r1)
    );


    R2FFT #(
        .FFT_LENGTH(FFT_LENGTH),
        .FFT_DW(FFT_DW),
        .PL_DEPTH(PL_DEPTH)
    ) u_fft (
        .clk(clk),
        .rst(rst),

        .autorun(autorun),
        .run(run),
        .fin(fin),
        .ifft(ifft),

        .done(done),
        .status(status),
        .bfpexp(bfpexp),

        .sact_istream(sact),
        .sdw_istream_real(sreal),
        .sdw_istream_imag(simag),

        .dmaact(dmaact),
        .dmaa(dmaa),
        .dmadr_real(dmadr_real),
        .dmadr_imag(dmadr_imag),

        .twact(twact),
        .twa(twa),
        .twdr_cos(twdr_cos),

        .ract_ram0(ract0),
        .ra_ram0(ra0),
        .rdr_ram0(r0),

        .wact_ram0(wact0),
        .wa_ram0(wa0),
        .wdw_ram0(w0),

        .ract_ram1(ract1),
        .ra_ram1(ra1),
        .rdr_ram1(r1),

        .wact_ram1(wact1),
        .wa_ram1(wa1),
        .wdw_ram1(w1)
    );

endmodule