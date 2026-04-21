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

    reg               b_sact;
    reg signed [15:0] b_sdw_real, b_sdw_imag;

    always @(posedge i_clk) begin
        b_sact     <= b_win_ce;
        b_sdw_real <= {{(16-IW){b_win_samp[IW-1]}}, b_win_samp};
        b_sdw_imag <= 16'd0;
    end

    wire              b_done_w;
    wire [2:0]        b_status_w;
    wire signed [7:0] b_bfpexp_w;
    reg               b_done_r;
    reg  signed [7:0] b_bfpexp_r;

    reg [FFT_N-1:0] b_dma_addr;
    reg             b_dmaact, b_dmaact_r;
    reg [FFT_N-1:0] b_dmaa_r;
    reg             b_readout_done, b_fin_r;

    wire signed [15:0] b_dmadr_real_w, b_dmadr_imag_w;
    reg  signed [15:0] b_dmadr_real_r, b_dmadr_imag_r;

    always @(posedge i_clk) begin
        b_done_r       <= b_done_w;
        b_bfpexp_r     <= b_bfpexp_w;
        b_dmaact_r     <= b_dmaact;
        b_dmaa_r       <= b_dma_addr;
        b_dmadr_real_r <= b_dmadr_real_w;
        b_dmadr_imag_r <= b_dmadr_imag_w;
        b_fin_r        <= b_readout_done;
    end

    reg [2*OW-1:0] b_result;
    reg            b_sync;

    always @(posedge i_clk) begin
        if (i_reset) begin
            b_dmaact <= 1'b0; b_dma_addr <= {FFT_N{1'b0}};
            b_readout_done <= 1'b0;
            b_sync <= 1'b0; b_result <= {2*OW{1'b0}};
        end else begin
            b_sync <= 1'b0;
            if (b_done_r && !b_readout_done && !b_dmaact) begin
                b_dmaact   <= 1'b1;
                b_dma_addr <= {FFT_N{1'b0}};
                b_sync     <= 1'b1;
            end else if (b_dmaact) begin
                b_result <= {
                    {{(OW-16){b_dmadr_real_r[15]}}, b_dmadr_real_r},
                    {{(OW-16){b_dmadr_imag_r[15]}}, b_dmadr_imag_r}
                };
                b_dma_addr <= b_dma_addr + 1'b1;
                if (b_dma_addr == FFT_SIZE - 1) begin
                    b_dmaact       <= 1'b0;
                    b_readout_done <= 1'b1;
                end
            end else if (b_readout_done) begin
                b_readout_done <= 1'b0;
            end
        end
    end

    wire [FFT_N-3:0] b_twa; wire b_twact; wire [15:0] b_twdr_cos;
    wire b_ract0, b_wact0; wire [FFT_N-2:0] b_ra0, b_wa0; wire [31:0] b_rdr0, b_wdw0;
    wire b_ract1, b_wact1; wire [FFT_N-2:0] b_ra1, b_wa1; wire [31:0] b_rdr1, b_wdw1;

    wire b_ram_active = b_ract0|b_wact0|b_ract1|b_wact1;
    wire b_fft_running = (b_status_w == 3'd3);
    reg  b_ram_active_r, b_fft_running_r;
    always @(posedge i_clk) begin
        b_ram_active_r  <= b_ram_active;
        b_fft_running_r <= b_fft_running;
    end
    wire b_next_stage = b_ram_active_r & ~b_ram_active & b_fft_running_r;

    fft_twiddle_rom u_twiddle_b (
        .clk(i_clk), .twact(b_twact), .twa(b_twa), .twdr_cos(b_twdr_cos)
    );
    fft_data_ram u_b_ram0 (
        .clk(i_clk), .rst(i_reset), .next_stage(b_next_stage),
        .ract(b_ract0), .ra(b_ra0), .rdata(b_rdr0),
        .wact(b_wact0), .wa(b_wa0), .wdata(b_wdw0)
    );
    fft_data_ram u_b_ram1 (
        .clk(i_clk), .rst(i_reset), .next_stage(b_next_stage),
        .ract(b_ract1), .ra(b_ra1), .rdata(b_rdr1),
        .wact(b_wact1), .wa(b_wa1), .wdata(b_wdw1)
    );
    R2FFT #(.FFT_LENGTH(FFT_SIZE), .FFT_DW(16), .PL_DEPTH(3)) u_r2fft_b (
        .clk(i_clk), .rst(rst),
        .autorun(1'b1), .run(1'b0), .fin(b_fin_r), .ifft(1'b0),
        .done(b_done_w), .status(b_status_w), .bfpexp(b_bfpexp_w),
        .sact_istream(b_sact),
        .sdw_istream_real(b_sdw_real), .sdw_istream_imag(b_sdw_imag),
        .dmaact(b_dmaact_r), .dmaa(b_dmaa_r),
        .dmadr_real(b_dmadr_real_w), .dmadr_imag(b_dmadr_imag_w),
        .twact(b_twact), .twa(b_twa), .twdr_cos(b_twdr_cos),
        .ract_ram0(b_ract0), .ra_ram0(b_ra0), .rdr_ram0(b_rdr0),
        .wact_ram0(b_wact0), .wa_ram0(b_wa0), .wdw_ram0(b_wdw0),
        .ract_ram1(b_ract1), .ra_ram1(b_ra1), .rdr_ram1(b_rdr1),
        .wact_ram1(b_wact1), .wa_ram1(b_wa1), .wdw_ram1(b_wdw1)
    );

endmodule