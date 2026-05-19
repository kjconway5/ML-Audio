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


    // Hanning window ROM (unsigned 16-bit, peak ~ 0xFFFF)
    reg [IW-1:0] hanning_rom [0:FFT_SIZE-1];
    initial $readmemh("hanning.hex", hanning_rom);

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
    //
    //   ST_IDLE     waits for frame_pending.
    //   ST_READOUT  streams FFT_SIZE windowed samples to the FFT,
    //               honouring its i_ready backpressure.
    //
    // read_addr_base latches wp_snapshot at the moment of transition,
    // so a same-cycle re-trigger that overwrites wp_snapshot doesn't
    // disturb the in-progress readout.
    localparam ST_IDLE    = 1'b0,
               ST_READOUT = 1'b1;

    reg              state;
    reg [FFT_N-1:0]  read_idx;
    reg [FFT_N-1:0]  read_addr_base;

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
    //
    // Note: read_addr_base + read_idx wraps mod FFT_SIZE naturally
    // because both registers are FFT_N bits wide. A simultaneous
    // input write to the same ring slot is benign — NBA semantics
    // mean the combinational read sees the *old* slot value, which
    // is exactly what the in-progress readout wants.
    wire [FFT_N-1:0]       read_addr   = read_addr_base + read_idx;
    wire signed [IW-1:0]   ring_sample = $signed(ring[read_addr]);
    wire signed [2*IW-1:0] product     = $signed(ring[read_addr])
                                       * $signed({1'b0, hanning_rom[read_idx]});
    wire signed [IW-1:0]   windowed    = product[2*IW-2 : IW-1];

    wire fft_i_valid = (state == ST_READOUT);

    // FFT instance (new ready/valid wrapper)

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