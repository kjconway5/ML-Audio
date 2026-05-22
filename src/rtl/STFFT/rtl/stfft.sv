`default_nettype none
module stfft #(
    parameter IW       = 16,
    parameter OW       = 16,
    parameter FFT_SIZE = 256,
    parameter FFT_N    = $clog2(FFT_SIZE),
    parameter HOP      = FFT_SIZE / 2
)(
    input  wire                     i_clk,
    input  wire                     i_reset,

    // Input axis  (producer drives i_valid/i_data; wrapper drives i_ready)
    input  wire                     i_valid,
    input  wire signed [IW-1:0]     i_data,
    output wire                     i_ready,

    // Output axis (wrapper drives o_valid/o_data/o_last; consumer drives o_ready)
    output wire                     o_valid,
    output wire signed [2*OW-1:0]   o_data,
    input  wire                     o_ready,
    output wire                     o_last,

    // Block-floating-point exponent for the currently-emitting frame
    output wire signed [7:0]        o_bfpexp
);

   
    // Hanning window coefficient

    // The 256-point Hanning window is symmetric around index 128:
    //   w[0]   = 0x0000      (boundary)
    //   w[n]   = w[256 - n]  for n in 1..127 (mirrors indices 129..255)
    //   w[128] = 0x7FFF      (peak — unique to itself)
    //

    function automatic [IW-1:0] hanning_coef(input [FFT_N-1:0] idx);
        reg [FFT_N-1:0] folded;
        begin
            // Fold idx >= 128 back into [0, 127] via 256 - idx.
            folded = idx[FFT_N-1] ? (~idx + 1'b1) : idx;

            if (idx == 8'd128) begin
                hanning_coef = 16'h7FFF;          // peak (unique)
            end else begin
                case (folded[FFT_N-2:0])
                    7'd0:   hanning_coef = 16'h0000;
                    7'd1:   hanning_coef = 16'h0005;
                    7'd2:   hanning_coef = 16'h0014;
                    7'd3:   hanning_coef = 16'h002C;
                    7'd4:   hanning_coef = 16'h004F;
                    7'd5:   hanning_coef = 16'h007B;
                    7'd6:   hanning_coef = 16'h00B1;
                    7'd7:   hanning_coef = 16'h00F1;
                    7'd8:   hanning_coef = 16'h013B;
                    7'd9:   hanning_coef = 16'h018E;
                    7'd10:  hanning_coef = 16'h01EB;
                    7'd11:  hanning_coef = 16'h0251;
                    7'd12:  hanning_coef = 16'h02C1;
                    7'd13:  hanning_coef = 16'h033B;
                    7'd14:  hanning_coef = 16'h03BE;
                    7'd15:  hanning_coef = 16'h044A;
                    7'd16:  hanning_coef = 16'h04DF;
                    7'd17:  hanning_coef = 16'h057E;
                    7'd18:  hanning_coef = 16'h0625;
                    7'd19:  hanning_coef = 16'h06D5;
                    7'd20:  hanning_coef = 16'h078F;
                    7'd21:  hanning_coef = 16'h0850;
                    7'd22:  hanning_coef = 16'h091B;
                    7'd23:  hanning_coef = 16'h09EE;
                    7'd24:  hanning_coef = 16'h0AC9;
                    7'd25:  hanning_coef = 16'h0BAD;
                    7'd26:  hanning_coef = 16'h0C98;
                    7'd27:  hanning_coef = 16'h0D8C;
                    7'd28:  hanning_coef = 16'h0E87;
                    7'd29:  hanning_coef = 16'h0F8A;
                    7'd30:  hanning_coef = 16'h1094;
                    7'd31:  hanning_coef = 16'h11A6;
                    7'd32:  hanning_coef = 16'h12BF;
                    7'd33:  hanning_coef = 16'h13DE;
                    7'd34:  hanning_coef = 16'h1505;
                    7'd35:  hanning_coef = 16'h1632;
                    7'd36:  hanning_coef = 16'h1766;
                    7'd37:  hanning_coef = 16'h18A0;
                    7'd38:  hanning_coef = 16'h19E0;
                    7'd39:  hanning_coef = 16'h1B26;
                    7'd40:  hanning_coef = 16'h1C71;
                    7'd41:  hanning_coef = 16'h1DC2;
                    7'd42:  hanning_coef = 16'h1F19;
                    7'd43:  hanning_coef = 16'h2074;
                    7'd44:  hanning_coef = 16'h21D4;
                    7'd45:  hanning_coef = 16'h2339;
                    7'd46:  hanning_coef = 16'h24A3;
                    7'd47:  hanning_coef = 16'h2610;
                    7'd48:  hanning_coef = 16'h2782;
                    7'd49:  hanning_coef = 16'h28F7;
                    7'd50:  hanning_coef = 16'h2A70;
                    7'd51:  hanning_coef = 16'h2BEC;
                    7'd52:  hanning_coef = 16'h2D6C;
                    7'd53:  hanning_coef = 16'h2EEE;
                    7'd54:  hanning_coef = 16'h3073;
                    7'd55:  hanning_coef = 16'h31FA;
                    7'd56:  hanning_coef = 16'h3383;
                    7'd57:  hanning_coef = 16'h350F;
                    7'd58:  hanning_coef = 16'h369C;
                    7'd59:  hanning_coef = 16'h382A;
                    7'd60:  hanning_coef = 16'h39BA;
                    7'd61:  hanning_coef = 16'h3B4A;
                    7'd62:  hanning_coef = 16'h3CDC;
                    7'd63:  hanning_coef = 16'h3E6D;
                    7'd64:  hanning_coef = 16'h3FFF;
                    7'd65:  hanning_coef = 16'h4192;
                    7'd66:  hanning_coef = 16'h4323;
                    7'd67:  hanning_coef = 16'h44B5;
                    7'd68:  hanning_coef = 16'h4645;
                    7'd69:  hanning_coef = 16'h47D5;
                    7'd70:  hanning_coef = 16'h4963;
                    7'd71:  hanning_coef = 16'h4AF0;
                    7'd72:  hanning_coef = 16'h4C7C;
                    7'd73:  hanning_coef = 16'h4E05;
                    7'd74:  hanning_coef = 16'h4F8C;
                    7'd75:  hanning_coef = 16'h5111;
                    7'd76:  hanning_coef = 16'h5293;
                    7'd77:  hanning_coef = 16'h5413;
                    7'd78:  hanning_coef = 16'h558F;
                    7'd79:  hanning_coef = 16'h5708;
                    7'd80:  hanning_coef = 16'h587D;
                    7'd81:  hanning_coef = 16'h59EF;
                    7'd82:  hanning_coef = 16'h5B5C;
                    7'd83:  hanning_coef = 16'h5CC6;
                    7'd84:  hanning_coef = 16'h5E2B;
                    7'd85:  hanning_coef = 16'h5F8B;
                    7'd86:  hanning_coef = 16'h60E6;
                    7'd87:  hanning_coef = 16'h623D;
                    7'd88:  hanning_coef = 16'h638E;
                    7'd89:  hanning_coef = 16'h64D9;
                    7'd90:  hanning_coef = 16'h661F;
                    7'd91:  hanning_coef = 16'h675F;
                    7'd92:  hanning_coef = 16'h6899;
                    7'd93:  hanning_coef = 16'h69CD;
                    7'd94:  hanning_coef = 16'h6AFA;
                    7'd95:  hanning_coef = 16'h6C21;
                    7'd96:  hanning_coef = 16'h6D40;
                    7'd97:  hanning_coef = 16'h6E59;
                    7'd98:  hanning_coef = 16'h6F6B;
                    7'd99:  hanning_coef = 16'h7075;
                    7'd100: hanning_coef = 16'h7178;
                    7'd101: hanning_coef = 16'h7273;
                    7'd102: hanning_coef = 16'h7367;
                    7'd103: hanning_coef = 16'h7452;
                    7'd104: hanning_coef = 16'h7536;
                    7'd105: hanning_coef = 16'h7611;
                    7'd106: hanning_coef = 16'h76E4;
                    7'd107: hanning_coef = 16'h77AF;
                    7'd108: hanning_coef = 16'h7870;
                    7'd109: hanning_coef = 16'h792A;
                    7'd110: hanning_coef = 16'h79DA;
                    7'd111: hanning_coef = 16'h7A81;
                    7'd112: hanning_coef = 16'h7B20;
                    7'd113: hanning_coef = 16'h7BB5;
                    7'd114: hanning_coef = 16'h7C41;
                    7'd115: hanning_coef = 16'h7CC4;
                    7'd116: hanning_coef = 16'h7D3E;
                    7'd117: hanning_coef = 16'h7DAE;
                    7'd118: hanning_coef = 16'h7E14;
                    7'd119: hanning_coef = 16'h7E71;
                    7'd120: hanning_coef = 16'h7EC4;
                    7'd121: hanning_coef = 16'h7F0E;
                    7'd122: hanning_coef = 16'h7F4E;
                    7'd123: hanning_coef = 16'h7F84;
                    7'd124: hanning_coef = 16'h7FB0;
                    7'd125: hanning_coef = 16'h7FD3;
                    7'd126: hanning_coef = 16'h7FEB;
                    7'd127: hanning_coef = 16'h7FFA;
                    default: hanning_coef = 16'h0000;
                endcase
            end
        end
    endfunction


    // Sliding-window ring buffer
    //
    // wp = next slot to write (= oldest currently-held slot). After a
    // write, that slot now holds the newest sample and wp advances.
    // The window from oldest to newest is ring[wp], ring[wp+1], ...,
    // ring[wp + FFT_SIZE - 1] (all mod FFT_SIZE).
    reg [IW-1:0]    ring [0:FFT_SIZE-1];
    reg [FFT_N-1:0] wp;

    reg [FFT_N-1:0] warmup_cnt;       // counts up to FFT_SIZE-1 then latches
    reg [FFT_N-1:0] hop_cnt;          // 0..HOP-1, used after warm-up
    reg             warmup_done;

    reg             frame_pending;    // 1-deep trigger queue
    reg [FFT_N-1:0] wp_snapshot;      // start address for the pending readout

    // Forward declaration so i_ready can see start_readout.
    wire            start_readout;

    wire input_xfer    = i_valid && i_ready;
    wire would_trigger = (!warmup_done && warmup_cnt == FFT_SIZE - 1) ||
                         ( warmup_done && hop_cnt    == HOP      - 1);

    // Forward-declare state machine signals (defined below) — needed here so
    // i_ready can reference them.
    wire [$clog2(FFT_SIZE)-1:0] read_idx_w;
    wire                        in_readout;

    // Ring corruption guard: during a readout, the read pointer advances
    // through slots base..base+255. New writes go to wp, which starts at
    // `read_addr_base` (= the oldest slot we're about to read). If the FFT
    // stalls and `read_idx` doesn't advance, wp will catch up and overwrite
    // slots we haven't read yet. Block writes whenever the next input would
    // overwrite a still-unread slot.
    //
    // wp_off counts writes since the start of the current readout:
    //   wp_off = (wp - read_addr_base) mod FFT_SIZE
    // Safe condition: wp_off < read_idx (we've already read past where the
    // next write would land). At the very first readout cycle wp_off=0 and
    // read_idx=0, so this blocks for one cycle until read_idx advances —
    // benign because the input rate is far below 1/cycle.
    reg [FFT_N-1:0]  read_addr_base;
    wire [FFT_N-1:0] wp_off       = wp - read_addr_base;
    wire             would_corrupt = in_readout && (wp_off >= read_idx_w);

    // i_ready drops if accepting this sample would EITHER queue a second
    // trigger before the first started OR overwrite an unread ring slot.
    assign i_ready = !(would_trigger && frame_pending && !start_readout) &&
                     !would_corrupt;

    integer i;
    always @(posedge i_clk) begin
        if (i_reset) begin
            wp            <= '0;
            warmup_cnt    <= '0;
            hop_cnt       <= '0;
            warmup_done   <= 1'b0;
            frame_pending <= 1'b0;
            wp_snapshot   <= '0;
        end else begin
            // The clear (start_readout) may be overridden in the same
            // cycle by the input-xfer set below — that's intentional:
            // a simultaneous trigger queues onto an already-starting
            // readout. The just-starting readout has its own latched
            // wp_snapshot copy in read_addr_base, so updating
            // wp_snapshot now affects only the *next* frame.
            if (start_readout) frame_pending <= 1'b0;

            if (input_xfer) begin
                ring[wp] <= i_data;
                wp       <= wp + 1'b1;

                if (!warmup_done) begin
                    warmup_cnt <= warmup_cnt + 1'b1;
                    if (warmup_cnt == FFT_SIZE - 1) begin
                        warmup_done   <= 1'b1;
                        hop_cnt       <= '0;
                        frame_pending <= 1'b1;
                        wp_snapshot   <= wp + 1'b1;
                    end
                end else begin
                    hop_cnt <= hop_cnt + 1'b1;
                    if (hop_cnt == HOP - 1) begin
                        hop_cnt       <= '0;
                        frame_pending <= 1'b1;
                        wp_snapshot   <= wp + 1'b1;
                    end
                end
            end
        end
    end


    // Frame readout state machine
    //   ST_IDLE     waits for frame_pending.
    //   ST_READOUT  streams FFT_SIZE windowed samples to the FFT,
    //               honouring its i_ready backpressure.


    localparam ST_IDLE    = 1'b0,
               ST_READOUT = 1'b1;

    reg              state;
    reg [FFT_N-1:0]  read_idx;

    assign start_readout = (state == ST_IDLE) && frame_pending;
    assign read_idx_w    = read_idx;
    assign in_readout    = (state == ST_READOUT);

    wire             fft_i_ready_w;
    wire             windowed_xfer = (state == ST_READOUT) && fft_i_ready_w;

    always @(posedge i_clk) begin
        if (i_reset) begin
            state          <= ST_IDLE;
            read_idx       <= '0;
            read_addr_base <= '0;
        end else begin
            case (state)
                ST_IDLE: begin
                    if (start_readout) begin
                        state          <= ST_READOUT;
                        read_idx       <= '0;
                        read_addr_base <= wp_snapshot;
                    end
                end
                ST_READOUT: begin
                    if (windowed_xfer) begin
                        if (read_idx == FFT_SIZE - 1)
                            state <= ST_IDLE;
                        else
                            read_idx <= read_idx + 1'b1;
                    end
                end
            endcase
        end
    end


    // Combinational ring read + Hanning window multiply

    wire [FFT_N-1:0]       read_addr   = read_addr_base + read_idx;
    wire signed [IW-1:0]   ring_sample = $signed(ring[read_addr]);
    wire        [IW-1:0]   hann_coef   = hanning_coef(read_idx);
    wire signed [2*IW-1:0] product     = $signed(ring[read_addr])
                                       * $signed({1'b0, hann_coef});
    wire signed [IW-1:0]   windowed    = product[2*IW-2 : IW-1];

    wire fft_i_valid = (state == ST_READOUT);


    // FFT
    fft #(
        .IW       (IW),
        .OW       (OW),
        .FFT_SIZE (FFT_SIZE)
    ) u_fft (
        .i_clk    (i_clk),
        .i_reset  (i_reset),

        .i_valid  (fft_i_valid),
        .i_data   (windowed),
        .i_ready  (fft_i_ready_w),

        .o_valid  (o_valid),
        .o_data   (o_data),
        .o_ready  (o_ready),
        .o_last   (o_last),

        .o_bfpexp (o_bfpexp)
    );

    // Silence unused-wire warnings under verilator on ring_sample
    // (it's the readable form of the multiply input — kept for clarity).
    /* verilator lint_off UNUSED */
    wire _unused_ring_sample = |ring_sample;
    /* verilator lint_on UNUSED */

endmodule