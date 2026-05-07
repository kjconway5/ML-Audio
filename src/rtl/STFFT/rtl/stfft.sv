// stfft.sv  (updated: fft_data_ram address width 7->8 bits)
//
// Changes vs original:
//   - Four fft_data_ram instantiations: ra/wa zero-extended from [FFT_N-2:0]
//     (7-bit) to [7:0] (8-bit) to match the updated fft_data_ram which uses
//     gf180mcu_ocd_ip_sram__sram256x8m8wm1 (3.3V, 256-entry, 8-bit addr).
//   - All other logic is identical to the incoming stfft.sv.
//
// Root cause: R2FFT #(.FFT_LENGTH(256)) drives ra_ram0/wa_ram0 as [6:0]
// because each of its two RAM banks only holds FFT_SIZE/2=128 entries.
// The 256-entry SRAM has a wider address bus; the top bit is unused and
// tied to 0 via the concatenation {1'b0, ra}.

`default_nettype none

module stfft #(
    parameter IW         = 16,
    parameter OW         = 16,
    parameter FFT_SIZE   = 256,
    parameter FFT_N      = $clog2(FFT_SIZE),
    parameter HOP        = FFT_SIZE / 2,
    parameter FIFO_DEPTH = 64,
    parameter FIFO_AW    = $clog2(FIFO_DEPTH)
)(
    input  wire              i_clk,
    input  wire              i_reset,
    input  wire              i_ce,
    input  wire [IW-1:0]     i_sample,
    output reg  [2*OW-1:0]   o_fft_result,
    output reg               o_fft_sync,
    output wire              win_ce_o,
    output wire signed [7:0] o_bfpexp
);

    reg rst;
    always @(posedge i_clk) rst <= i_reset;

    reg [IW-1:0] hanning_rom [0:FFT_SIZE-1];
    initial $readmemh("hanning.hex", hanning_rom);

    // Global sample counter (b_armed trigger only)
    reg [FFT_N-1:0] sample_cnt;
    always @(posedge i_clk) begin
        if (i_reset)   sample_cnt <= {FFT_N{1'b0}};
        else if (i_ce) sample_cnt <= sample_cnt + 1'b1;
    end

    reg b_armed;
    always @(posedge i_clk) begin
        if (i_reset)
            b_armed <= 1'b0;
        else if (!b_armed && i_ce && (sample_cnt == HOP[FFT_N-1:0]))
            b_armed <= 1'b1;
    end

    // Channel A

    reg [IW-1:0]    a_fifo_mem [0:FIFO_DEPTH-1];
    reg [FIFO_AW:0] a_fifo_wp, a_fifo_rp, a_fifo_cnt;
    wire a_fifo_empty = (a_fifo_cnt == 0);
    wire a_fifo_full  = (a_fifo_cnt == FIFO_DEPTH[FIFO_AW:0]);

    wire [2:0] a_status_w;
    wire       a_in_stream = (a_status_w == 3'd1);

    reg  [FFT_N:0] a_drain_cnt;
    wire           a_can_drain = (a_drain_cnt < FFT_SIZE[FFT_N:0]);
    wire           a_do_drain  = a_in_stream && !a_fifo_empty && a_can_drain;

    always @(posedge i_clk) begin
        if (i_reset || !a_in_stream) a_drain_cnt <= 0;
        else if (a_do_drain)         a_drain_cnt <= a_drain_cnt + 1'b1;
    end

    always @(posedge i_clk) begin
        if (i_reset) begin
            a_fifo_wp <= 0; a_fifo_rp <= 0; a_fifo_cnt <= 0;
        end else begin
            if (i_ce && !a_fifo_full) begin
                a_fifo_mem[a_fifo_wp[FIFO_AW-1:0]] <= i_sample;
                a_fifo_wp <= a_fifo_wp + 1'b1;
            end
            if (a_do_drain) a_fifo_rp <= a_fifo_rp + 1'b1;
            case ({i_ce && !a_fifo_full, a_do_drain})
                2'b10: a_fifo_cnt <= a_fifo_cnt + 1'b1;
                2'b01: a_fifo_cnt <= a_fifo_cnt - 1'b1;
                default: ;
            endcase
        end
    end
    wire [IW-1:0] a_fifo_out = a_fifo_mem[a_fifo_rp[FIFO_AW-1:0]];

    reg [FFT_N-1:0] a_hann_idx;
    always @(posedge i_clk) begin
        if (i_reset || !a_in_stream) a_hann_idx <= 0;
        else if (a_do_drain)         a_hann_idx <= a_hann_idx + 1'b1;
    end

    wire signed [2*IW-1:0] a_prod_w =
        $signed(a_fifo_out) * $signed({1'b0, hanning_rom[a_hann_idx]});
    wire signed [IW-1:0] a_win_w = a_prod_w[2*IW-2:IW-1];

    reg signed [IW-1:0] a_win;
    reg                 a_win_ce;
    always @(posedge i_clk) begin
        if (i_reset) begin a_win <= 0; a_win_ce <= 1'b0; end
        else         begin a_win <= a_win_w; a_win_ce <= a_do_drain; end
    end

    reg               a_sact;
    reg signed [15:0] a_sdw_real, a_sdw_imag;
    always @(posedge i_clk) begin
        a_sact     <= a_win_ce;
        a_sdw_real <= {{(16-IW){a_win[IW-1]}}, a_win};
        a_sdw_imag <= 16'd0;
    end

    wire              a_done_w;
    wire signed [7:0] a_bfpexp_w;
    reg               a_done_r;
    reg  signed [7:0] a_bfpexp_r;
    reg               a_done_ack;
    reg [FFT_N-1:0]   a_dma_addr;
    reg               a_dmaact, a_dmaact_r;
    reg [FFT_N-1:0]   a_dmaa_r;
    reg               a_readout_done, a_fin_r;
    wire signed [15:0] a_dmadr_real_w, a_dmadr_imag_w;
    reg  signed [15:0] a_dmadr_real_r, a_dmadr_imag_r;

    always @(posedge i_clk) begin
        a_done_r <= a_done_w; a_bfpexp_r <= a_bfpexp_w;
        a_dmaact_r <= a_dmaact; a_dmaa_r <= a_dma_addr;
        a_dmadr_real_r <= a_dmadr_real_w; a_dmadr_imag_r <= a_dmadr_imag_w;
        a_fin_r <= a_readout_done;
    end
    always @(posedge i_clk) begin
        if (i_reset)       a_done_ack <= 1'b0;
        else if (!a_done_r) a_done_ack <= 1'b0;
        else if (a_dmaact)  a_done_ack <= 1'b1;
    end

    reg [2*OW-1:0] a_result;
    reg            a_sync;
    always @(posedge i_clk) begin
        if (i_reset) begin
            a_dmaact<=1'b0; a_dma_addr<={FFT_N{1'b0}};
            a_readout_done<=1'b0; a_sync<=1'b0; a_result<={2*OW{1'b0}};
        end else begin
            a_sync <= 1'b0;
            if (a_done_r && !a_done_ack && !a_readout_done && !a_dmaact) begin
                a_dmaact <= 1'b1; a_dma_addr <= {FFT_N{1'b0}}; a_sync <= 1'b1;
            end else if (a_dmaact) begin
                a_result <= {{{(OW-16){a_dmadr_real_r[15]}},a_dmadr_real_r},
                             {{(OW-16){a_dmadr_imag_r[15]}},a_dmadr_imag_r}};
                a_dma_addr <= a_dma_addr + 1'b1;
                if (a_dma_addr == FFT_SIZE-1) begin
                    a_dmaact <= 1'b0; a_readout_done <= 1'b1;
                end
            end else if (a_readout_done)
                a_readout_done <= 1'b0;
        end
    end

    wire [FFT_N-3:0] a_twa; wire a_twact; wire [15:0] a_twdr_cos;
    wire a_ract0,a_wact0; wire [FFT_N-2:0] a_ra0,a_wa0; wire [31:0] a_rdr0,a_wdw0;
    wire a_ract1,a_wact1; wire [FFT_N-2:0] a_ra1,a_wa1; wire [31:0] a_rdr1,a_wdw1;

    wire a_ram_active = a_ract0|a_wact0|a_ract1|a_wact1;
    wire a_fft_running = (a_status_w == 3'd3);
    reg  a_ram_r, a_fft_r;
    always @(posedge i_clk) begin a_ram_r<=a_ram_active; a_fft_r<=a_fft_running; end
    wire a_next_stage = a_ram_r & ~a_ram_active & a_fft_r;

    fft_twiddle_rom u_twiddle_a (.clk(i_clk),.twact(a_twact),.twa(a_twa),.twdr_cos(a_twdr_cos));

    // CHANGED: {1'b0, a_ra0} and {1'b0, a_wa0}  -- zero-extend 7->8 bits
    fft_data_ram u_a_ram0 (.clk(i_clk),.rst(i_reset),.next_stage(a_next_stage),
        .ract(a_ract0),.ra({1'b0,a_ra0}),.rdata(a_rdr0),
        .wact(a_wact0),.wa({1'b0,a_wa0}),.wdata(a_wdw0));
    fft_data_ram u_a_ram1 (.clk(i_clk),.rst(i_reset),.next_stage(a_next_stage),
        .ract(a_ract1),.ra({1'b0,a_ra1}),.rdata(a_rdr1),
        .wact(a_wact1),.wa({1'b0,a_wa1}),.wdata(a_wdw1));

    R2FFT #(.FFT_LENGTH(FFT_SIZE),.FFT_DW(16),.PL_DEPTH(3)) u_r2fft_a (
        .clk(i_clk),.rst(rst),.autorun(1'b1),.run(1'b0),.fin(a_fin_r),.ifft(1'b0),
        .done(a_done_w),.status(a_status_w),.bfpexp(a_bfpexp_w),
        .sact_istream(a_sact),.sdw_istream_real(a_sdw_real),.sdw_istream_imag(a_sdw_imag),
        .dmaact(a_dmaact_r),.dmaa(a_dmaa_r),.dmadr_real(a_dmadr_real_w),.dmadr_imag(a_dmadr_imag_w),
        .twact(a_twact),.twa(a_twa),.twdr_cos(a_twdr_cos),
        .ract_ram0(a_ract0),.ra_ram0(a_ra0),.rdr_ram0(a_rdr0),
        .wact_ram0(a_wact0),.wa_ram0(a_wa0),.wdw_ram0(a_wdw0),
        .ract_ram1(a_ract1),.ra_ram1(a_ra1),.rdr_ram1(a_rdr1),
        .wact_ram1(a_wact1),.wa_ram1(a_wa1),.wdw_ram1(a_wdw1));

    // Channel B

    reg [IW-1:0]    b_fifo_mem [0:FIFO_DEPTH-1];
    reg [FIFO_AW:0] b_fifo_wp, b_fifo_rp, b_fifo_cnt;
    wire b_fifo_empty = (b_fifo_cnt == 0);
    wire b_fifo_full  = (b_fifo_cnt == FIFO_DEPTH[FIFO_AW:0]);

    wire [2:0] b_status_w;
    wire       b_in_stream = (b_status_w == 3'd1);

    reg  [FFT_N:0] b_drain_cnt;
    wire           b_can_drain = (b_drain_cnt < FFT_SIZE[FFT_N:0]);
    wire           b_do_drain  = b_in_stream && !b_fifo_empty && b_can_drain;

    always @(posedge i_clk) begin
        if (i_reset || !b_in_stream) b_drain_cnt <= 0;
        else if (b_do_drain)         b_drain_cnt <= b_drain_cnt + 1'b1;
    end

    always @(posedge i_clk) begin
        if (i_reset) begin
            b_fifo_wp <= 0; b_fifo_rp <= 0; b_fifo_cnt <= 0;
        end else begin
            if (i_ce && b_armed && !b_fifo_full) begin
                b_fifo_mem[b_fifo_wp[FIFO_AW-1:0]] <= i_sample;
                b_fifo_wp <= b_fifo_wp + 1'b1;
            end
            if (b_do_drain) b_fifo_rp <= b_fifo_rp + 1'b1;
            case ({i_ce && b_armed && !b_fifo_full, b_do_drain})
                2'b10: b_fifo_cnt <= b_fifo_cnt + 1'b1;
                2'b01: b_fifo_cnt <= b_fifo_cnt - 1'b1;
                default: ;
            endcase
        end
    end
    wire [IW-1:0] b_fifo_out = b_fifo_mem[b_fifo_rp[FIFO_AW-1:0]];

    reg [FFT_N-1:0] b_hann_idx;
    always @(posedge i_clk) begin
        if (i_reset || !b_in_stream) b_hann_idx <= 0;
        else if (b_do_drain)         b_hann_idx <= b_hann_idx + 1'b1;
    end

    wire signed [2*IW-1:0] b_prod_w =
        $signed(b_fifo_out) * $signed({1'b0, hanning_rom[b_hann_idx]});
    wire signed [IW-1:0] b_win_w = b_prod_w[2*IW-2:IW-1];

    reg signed [IW-1:0] b_win;
    reg                 b_win_ce;
    always @(posedge i_clk) begin
        if (i_reset) begin b_win <= 0; b_win_ce <= 1'b0; end
        else         begin b_win <= b_win_w; b_win_ce <= b_do_drain; end
    end

    reg               b_sact;
    reg signed [15:0] b_sdw_real, b_sdw_imag;
    always @(posedge i_clk) begin
        b_sact     <= b_win_ce;
        b_sdw_real <= {{(16-IW){b_win[IW-1]}}, b_win};
        b_sdw_imag <= 16'd0;
    end

    wire              b_done_w;
    wire signed [7:0] b_bfpexp_w;
    reg               b_done_r;
    reg  signed [7:0] b_bfpexp_r;
    reg               b_done_ack;
    reg [FFT_N-1:0]   b_dma_addr;
    reg               b_dmaact, b_dmaact_r;
    reg [FFT_N-1:0]   b_dmaa_r;
    reg               b_readout_done, b_fin_r;
    wire signed [15:0] b_dmadr_real_w, b_dmadr_imag_w;
    reg  signed [15:0] b_dmadr_real_r, b_dmadr_imag_r;

    always @(posedge i_clk) begin
        b_done_r <= b_done_w; b_bfpexp_r <= b_bfpexp_w;
        b_dmaact_r <= b_dmaact; b_dmaa_r <= b_dma_addr;
        b_dmadr_real_r <= b_dmadr_real_w; b_dmadr_imag_r <= b_dmadr_imag_w;
        b_fin_r <= b_readout_done;
    end
    always @(posedge i_clk) begin
        if (i_reset)        b_done_ack <= 1'b0;
        else if (!b_done_r)  b_done_ack <= 1'b0;
        else if (b_dmaact)   b_done_ack <= 1'b1;
    end

    reg [2*OW-1:0] b_result;
    reg            b_sync;
    always @(posedge i_clk) begin
        if (i_reset) begin
            b_dmaact<=1'b0; b_dma_addr<={FFT_N{1'b0}};
            b_readout_done<=1'b0; b_sync<=1'b0; b_result<={2*OW{1'b0}};
        end else begin
            b_sync <= 1'b0;
            if (b_done_r && !b_done_ack && !b_readout_done && !b_dmaact) begin
                b_dmaact <= 1'b1; b_dma_addr <= {FFT_N{1'b0}}; b_sync <= 1'b1;
            end else if (b_dmaact) begin
                b_result <= {{{(OW-16){b_dmadr_real_r[15]}},b_dmadr_real_r},
                             {{(OW-16){b_dmadr_imag_r[15]}},b_dmadr_imag_r}};
                b_dma_addr <= b_dma_addr + 1'b1;
                if (b_dma_addr == FFT_SIZE-1) begin
                    b_dmaact <= 1'b0; b_readout_done <= 1'b1;
                end
            end else if (b_readout_done)
                b_readout_done <= 1'b0;
        end
    end

    wire [FFT_N-3:0] b_twa; wire b_twact; wire [15:0] b_twdr_cos;
    wire b_ract0,b_wact0; wire [FFT_N-2:0] b_ra0,b_wa0; wire [31:0] b_rdr0,b_wdw0;
    wire b_ract1,b_wact1; wire [FFT_N-2:0] b_ra1,b_wa1; wire [31:0] b_rdr1,b_wdw1;

    wire b_ram_active = b_ract0|b_wact0|b_ract1|b_wact1;
    wire b_fft_running = (b_status_w == 3'd3);
    reg  b_ram_r, b_fft_r;
    always @(posedge i_clk) begin b_ram_r<=b_ram_active; b_fft_r<=b_fft_running; end
    wire b_next_stage = b_ram_r & ~b_ram_active & b_fft_r;

    fft_twiddle_rom u_twiddle_b (.clk(i_clk),.twact(b_twact),.twa(b_twa),.twdr_cos(b_twdr_cos));

    // CHANGED: {1'b0, b_ra0} and {1'b0, b_wa0}  -- zero-extend 7->8 bits
    fft_data_ram u_b_ram0 (.clk(i_clk),.rst(i_reset),.next_stage(b_next_stage),
        .ract(b_ract0),.ra({1'b0,b_ra0}),.rdata(b_rdr0),
        .wact(b_wact0),.wa({1'b0,b_wa0}),.wdata(b_wdw0));
    fft_data_ram u_b_ram1 (.clk(i_clk),.rst(i_reset),.next_stage(b_next_stage),
        .ract(b_ract1),.ra({1'b0,b_ra1}),.rdata(b_rdr1),
        .wact(b_wact1),.wa({1'b0,b_wa1}),.wdata(b_wdw1));

    R2FFT #(.FFT_LENGTH(FFT_SIZE),.FFT_DW(16),.PL_DEPTH(3)) u_r2fft_b (
        .clk(i_clk),.rst(rst),.autorun(1'b1),.run(1'b0),.fin(b_fin_r),.ifft(1'b0),
        .done(b_done_w),.status(b_status_w),.bfpexp(b_bfpexp_w),
        .sact_istream(b_sact),.sdw_istream_real(b_sdw_real),.sdw_istream_imag(b_sdw_imag),
        .dmaact(b_dmaact_r),.dmaa(b_dmaa_r),.dmadr_real(b_dmadr_real_w),.dmadr_imag(b_dmadr_imag_w),
        .twact(b_twact),.twa(b_twa),.twdr_cos(b_twdr_cos),
        .ract_ram0(b_ract0),.ra_ram0(b_ra0),.rdr_ram0(b_rdr0),
        .wact_ram0(b_wact0),.wa_ram0(b_wa0),.wdw_ram0(b_wdw0),
        .ract_ram1(b_ract1),.ra_ram1(b_ra1),.rdr_ram1(b_rdr1),
        .wact_ram1(b_wact1),.wa_ram1(b_wa1),.wdw_ram1(b_wdw1));


    // Output MUX

    assign win_ce_o = a_win_ce | b_win_ce;

    always @(posedge i_clk) begin
        if (i_reset) begin
            o_fft_sync   <= 1'b0;
            o_fft_result <= {2*OW{1'b0}};
        end else begin
            o_fft_sync <= a_sync | b_sync;
            if      (a_dmaact) o_fft_result <= a_result;
            else if (b_dmaact) o_fft_result <= b_result;
        end
    end

    assign o_bfpexp = a_dmaact ? a_bfpexp_r : b_bfpexp_r;
    wire [2:0] debug_a_status = a_status_w;
    wire [2:0] debug_b_status = b_status_w;
    wire debug_a_in_stream = a_in_stream;
    wire debug_b_in_stream = b_in_stream;
    wire [8:0] debug_a_fifo_cnt = a_fifo_cnt;
    wire [8:0] debug_b_fifo_cnt = b_fifo_cnt;

endmodule