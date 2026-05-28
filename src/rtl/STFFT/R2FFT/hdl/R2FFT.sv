/*
 * Configurable Radix-2 FFT Processor with Block Floating-Point
 *
 * Variant: Ping-Pong + Single-Port RAM, with overlapped I/O and compute.
 *
 *   - RAM0 and RAM1 are each FFT_LENGTH deep, single-port (1 access/cycle).
 *   - Each RAM independently walks a lifecycle:
 *       IDLE → FILLING → READY_COMPUTE → COMPUTING → READY_DMA → DMAING → IDLE
 *     Compute uses whichever RAM is in COMPUTING. Stream writes go to
 *     whichever is FILLING. DMA reads come from whichever is DMAING. The
 *     two RAMs naturally pipeline in opposite phases so a new frame can be
 *     streamed in while the previous one is computing, and the previous
 *     frame's results can be DMA'd out while the current one is computing.
 *   - Per-frame BFP exponents are latched into per-RAM registers so each
 *     frame's exponent survives until DMA reads it out.
 *
 *   butterflyCore, bfp_*, bitReverseCounter, twiddleFactorRomBridge are
 *   reused unchanged from the original codebase.
 */

module R2FFT
  #(
    parameter FFT_LENGTH = 256,
    parameter FFT_DW     = 16,
    parameter PL_DEPTH   = 3,
    parameter FFT_N      = $clog2(FFT_LENGTH)
    )
   (
    // system
    input  wire                       clk,
    input  wire                       rst,

    // control (autorun=1 recommended for streaming; run/fin are legacy)
    input  wire                       autorun,
    input  wire                       run,
    input  wire                       fin,
    input  wire                       ifft,

    // status
    output wire                       done,    // a result is queued for DMA
    output wire [2:0]                 status,
    output wire signed [7:0]          bfpexp,
    output wire                       s_ready, // 1 when the FFT can accept an input sample

    // input stream
    input  wire                       sact_istream,
    input  wire signed [FFT_DW-1:0]   sdw_istream_real,
    input  wire signed [FFT_DW-1:0]   sdw_istream_imag,

    // DMA bus
    input  wire                       dmaact,
    input  wire [FFT_N-1:0]           dmaa,
    output wire signed [FFT_DW-1:0]   dmadr_real,
    output wire signed [FFT_DW-1:0]   dmadr_imag,

    // twiddle factor ROM
    output wire                       twact,
    output wire [FFT_N-1-2:0]         twa,
    input  wire [FFT_DW-1:0]          twdr_cos,

    // Single-port RAM 0
    output reg                        act_ram0,
    output reg                        we_ram0,
    output reg  [FFT_N-1:0]           a_ram0,
    output reg  [FFT_DW*2-1:0]        dw_ram0,
    input  wire [FFT_DW*2-1:0]        dr_ram0,

    // Single-port RAM 1
    output reg                        act_ram1,
    output reg                        we_ram1,
    output reg  [FFT_N-1:0]           a_ram1,
    output reg  [FFT_DW*2-1:0]        dw_ram1,
    input  wire [FFT_DW*2-1:0]        dr_ram1
    );

   // silence unused legacy ports (kept for interface compatibility)
   wire _unused_ok = &{1'b0, run, fin};

   localparam FFT_BFPDW       = $clog2(FFT_DW) + 1;
   localparam STAGE_COUNT_BW  = $clog2(FFT_N);


   // Per-RAM lifecycle state encoding (typedef + localparam, not enum,
   // so strict SV checkers don't require explicit casts on assignments).

   typedef logic [2:0] ram_state_t;
   localparam ram_state_t RAM_IDLE          = 3'd0;
   localparam ram_state_t RAM_FILLING       = 3'd1;
   localparam ram_state_t RAM_READY_COMPUTE = 3'd2;
   localparam ram_state_t RAM_COMPUTING     = 3'd3;
   localparam ram_state_t RAM_READY_DMA     = 3'd4;
   localparam ram_state_t RAM_DMAING        = 3'd5;

   ram_state_t ram0_state, ram1_state;
   reg         fill_next;  // 0 → next stream targets ram0; 1 → ram1

   // Convenience signals
   wire ram0_idle      = (ram0_state == RAM_IDLE);
   wire ram1_idle      = (ram1_state == RAM_IDLE);
   wire ram0_filling   = (ram0_state == RAM_FILLING);
   wire ram1_filling   = (ram1_state == RAM_FILLING);
   wire ram0_computing = (ram0_state == RAM_COMPUTING);
   wire ram1_computing = (ram1_state == RAM_COMPUTING);
   wire ram0_dmaing    = (ram0_state == RAM_DMAING);
   wire ram1_dmaing    = (ram1_state == RAM_DMAING);
   wire ram0_rcomp     = (ram0_state == RAM_READY_COMPUTE);
   wire ram1_rcomp     = (ram1_state == RAM_READY_COMPUTE);
   wire ram0_rdma      = (ram0_state == RAM_READY_DMA);
   wire ram1_rdma      = (ram1_state == RAM_READY_DMA);

   wire any_filling    = ram0_filling   || ram1_filling;
   wire any_computing  = ram0_computing || ram1_computing;
   wire any_dmaing     = ram0_dmaing    || ram1_dmaing;
   wire any_rdma       = ram0_rdma      || ram1_rdma;


   // Stream-arrival decision (combinational so first sample of each new
   // frame lands the same cycle we transition IDLE → FILLING).

   wire start_fill_ram0 = !any_filling && sact_istream && ram0_idle &&
                          (fill_next == 1'b0 || !ram1_idle);
   wire start_fill_ram1 = !any_filling && sact_istream && ram1_idle &&
                          (fill_next == 1'b1 || !ram0_idle);

   wire eff_filling_0   = ram0_filling || start_fill_ram0;
   wire eff_filling_1   = ram1_filling || start_fill_ram1;
   wire any_eff_filling = eff_filling_0 || eff_filling_1;


   // Bit-reverse input address counter
   //   Clears when nobody is filling (so each new frame starts at addr 0).
   //   Also clears explicitly on stream_done so back-to-back frames reset.

   wire [FFT_N-1:0] istreamAddr;
   wire             streamBufferFull;
   wire             stream_done;

   bitReverseCounter #(.BIT_WIDTH(FFT_N)) ubitReverseCounter
     (.rst       (rst),
      .clk       (clk),
      .clr       (!any_eff_filling || stream_done),
      .inc       (sact_istream && any_eff_filling),
      .iter      (istreamAddr),
      .count     (),
      .countFull (streamBufferFull));

   assign stream_done = sact_istream && streamBufferFull && any_eff_filling;


   // DMA byte counter — fires dma_done when last sample's address has been
   // presented. Uses dmaact-gated increments.

   reg [FFT_N-1:0] dma_counter;
   always @(posedge clk) begin
      if (rst || !any_dmaing) dma_counter <= '0;
      else if (dmaact)        dma_counter <= dma_counter + 1'b1;
   end
   wire dma_done = dmaact && any_dmaing && (dma_counter == (FFT_LENGTH-1));


   // BFP bit-width tracking on input stream — clears at the end of each
   // frame's stream so consecutive frames start fresh. The per-RAM init bw
   // registers latch the final (incl. last sample) max for each frame.

   wire [FFT_BFPDW-1:0] istreamBw;
   bfp_bitWidthDetector #(.FFT_BFPDW(FFT_BFPDW), .FFT_DW(FFT_DW))
     uistreamBitWidthDetector
     (.operand0 (sdw_istream_real),
      .operand1 (sdw_istream_imag),
      .operand2 ({FFT_DW{1'b0}}),
      .operand3 ({FFT_DW{1'b0}}),
      .bw       (istreamBw));

   wire [FFT_BFPDW-1:0] istreamMaxBw;
   bfp_maxBitWidth #(.FFT_BFPDW(FFT_BFPDW)) ubfp_maxBitWidthIstream
     (.rst    (rst),
      .clk    (clk),
      .clr    (!any_eff_filling || stream_done),
      .bw_act (sact_istream && any_eff_filling),
      .bw     (istreamBw),
      .max_bw (istreamMaxBw));

   // Combine the current cycle's bw with the running max so the LAST sample
   // of the frame is included even though the max register clears at
   // stream_done.
   wire [FFT_BFPDW-1:0] finalMaxBw =
        (istreamBw > istreamMaxBw) ? istreamBw : istreamMaxBw;

   reg [FFT_BFPDW-1:0] ram0_init_bw, ram1_init_bw;
   always @(posedge clk) begin
      if (rst) begin
         ram0_init_bw <= '0;
         ram1_init_bw <= '0;
      end else if (stream_done) begin
         if (eff_filling_0) ram0_init_bw <= finalMaxBw;
         if (eff_filling_1) ram1_init_bw <= finalMaxBw;
      end
   end


   // FFT sub-sequencer state encoding (typedef + localparam).

   typedef logic [2:0] sub_state_t;
   localparam sub_state_t SB_IDLE          = 3'd0;
   localparam sub_state_t SB_SETUP         = 3'd1;
   localparam sub_state_t SB_RUN           = 3'd2;
   localparam sub_state_t SB_WAIT_PIPELINE = 3'd3;
   localparam sub_state_t SB_NEXT_STAGE    = 3'd4;
   localparam sub_state_t SB_DONE          = 3'd5;

   sub_state_t sb_state_f, sb_state_n;


   // FFT sub-sequencer — runs while any RAM is COMPUTING.

   wire run_fft = any_computing;

   wire fin_fft = (sb_state_f == SB_DONE);

   localparam MAX_FFT_STAGE = FFT_N - 1;

   reg  [STAGE_COUNT_BW-1:0] fftStageCount;
   wire fftStageCountFull = (fftStageCount == MAX_FFT_STAGE);

   wire [FFT_BFPDW-1:0] currentBfpBw, nextBfpBw;
   wire signed [7:0]    currentBfpExp;
   wire                 iteratorDone;
   wire                 oactFftUnit;

   // Initial bw fed into the BFP accumulator at SB_SETUP — comes from whichever
   // RAM just entered COMPUTING.
   wire compute_target = ram1_computing ? 1'b1 : 1'b0;
   wire [FFT_BFPDW-1:0] initBwForCompute =
        compute_target ? ram1_init_bw : ram0_init_bw;

   bfp_bitWidthAcc #(.FFT_BFPDW(FFT_BFPDW), .FFT_DW(FFT_DW)) ubfpacc
     (.clk          (clk),
      .rst          (rst),
      .init         (sb_state_f == SB_SETUP),
      .bw_init      (initBwForCompute),
      .update       ((sb_state_f == SB_NEXT_STAGE) && !fftStageCountFull),
      .bw_new       (nextBfpBw),
      .bfp_bw       (currentBfpBw),
      .bfp_exponent (currentBfpExp));

   always_comb begin
      if (!run_fft) sb_state_n = SB_IDLE;
      else case (sb_state_f)
        SB_IDLE:           sb_state_n = SB_SETUP;
        SB_SETUP:          sb_state_n = SB_RUN;
        SB_RUN:            sb_state_n = iteratorDone ? SB_WAIT_PIPELINE : SB_RUN;
        SB_WAIT_PIPELINE:  sb_state_n = oactFftUnit  ? SB_WAIT_PIPELINE : SB_NEXT_STAGE;
        SB_NEXT_STAGE:     sb_state_n = fftStageCountFull ? SB_DONE : SB_RUN;
        SB_DONE:           sb_state_n = SB_DONE;
        default:           sb_state_n = SB_IDLE;
      endcase
   end

   always @(posedge clk) begin
      if (rst) sb_state_f <= SB_IDLE;
      else     sb_state_f <= sb_state_n;
   end

   always @(posedge clk) begin
      if (rst)
         fftStageCount <= '0;
      else case (sb_state_f)
        SB_IDLE, SB_SETUP: fftStageCount <= '0;
        SB_NEXT_STAGE:     fftStageCount <= fftStageCount + 1'b1;
        default: ;
      endcase
   end

   // Per-RAM final BFP exponent — captured at compute-done for the RAM that
   // just finished. The top-level bfpexp output reflects whichever RAM is
   // queued for DMA (READY_DMA or DMAING).
   reg signed [7:0] ram0_bfpexp, ram1_bfpexp;
   always @(posedge clk) begin
      if (rst) begin
         ram0_bfpexp <= '0;
         ram1_bfpexp <= '0;
      end else if (fin_fft) begin
         if (ram0_computing) ram0_bfpexp <= currentBfpExp;
         if (ram1_computing) ram1_bfpexp <= currentBfpExp;
      end
   end

   wire dma_target = (ram1_dmaing || ram1_rdma) ? 1'b1 : 1'b0;
   assign bfpexp   = dma_target ? ram1_bfpexp : ram0_bfpexp;


   // Butterfly engine — drives whichever RAM is COMPUTING

   wire                  bf_act;
   wire                  bf_we;
   wire [FFT_N-1:0]      bf_a;
   wire [FFT_DW*2-1:0]   bf_dw;
   wire [FFT_DW*2-1:0]   bf_dr = compute_target ? dr_ram1 : dr_ram0;

   butterflyUnit
     #(.FFT_N         (FFT_N),
       .FFT_DW        (FFT_DW),
       .FFT_BFPDW     (FFT_BFPDW),
       .PL_DEPTH      (PL_DEPTH),
       .STAGE_COUNT_BW(STAGE_COUNT_BW))
   ubutterflyUnit
     (.clk          (clk),
      .rst          (rst),

      .clr_bfp      (sb_state_f == SB_NEXT_STAGE),
      .ibfp         (currentBfpBw),
      .obfp         (nextBfpBw),

      .run          (sb_state_f == SB_RUN),
      .stageCount   (fftStageCount),
      .iteratorDone (iteratorDone),
      .oact         (oactFftUnit),
      .ifft         (ifft),

      .twact        (twact),
      .twa          (twa),
      .twdr_cos     (twdr_cos),

      .act          (bf_act),
      .we           (bf_we),
      .a            (bf_a),
      .dw           (bf_dw),
      .dr           (bf_dr));


   // RAM 0 port — combinational mux by ram0's effective state

   always_comb begin
      if (eff_filling_0) begin
         act_ram0 = sact_istream;
         we_ram0  = 1'b1;
         a_ram0   = istreamAddr;
         dw_ram0  = {sdw_istream_imag, sdw_istream_real};
      end else if (ram0_computing) begin
         act_ram0 = bf_act;
         we_ram0  = bf_we;
         a_ram0   = bf_a;
         dw_ram0  = bf_dw;
      end else if (ram0_dmaing) begin
         act_ram0 = dmaact;
         we_ram0  = 1'b0;
         a_ram0   = dmaa;
         dw_ram0  = '0;
      end else begin
         act_ram0 = 1'b0;
         we_ram0  = 1'b0;
         a_ram0   = '0;
         dw_ram0  = '0;
      end
   end


   // RAM 1 port

   always_comb begin
      if (eff_filling_1) begin
         act_ram1 = sact_istream;
         we_ram1  = 1'b1;
         a_ram1   = istreamAddr;
         dw_ram1  = {sdw_istream_imag, sdw_istream_real};
      end else if (ram1_computing) begin
         act_ram1 = bf_act;
         we_ram1  = bf_we;
         a_ram1   = bf_a;
         dw_ram1  = bf_dw;
      end else if (ram1_dmaing) begin
         act_ram1 = dmaact;
         we_ram1  = 1'b0;
         a_ram1   = dmaa;
         dw_ram1  = '0;
      end else begin
         act_ram1 = 1'b0;
         we_ram1  = 1'b0;
         a_ram1   = '0;
         dw_ram1  = '0;
      end
   end

   // DMA read data — from whichever RAM is currently DMAING. Falls back to
   // ram0 when idle (caller should only sample when dmaact has been high).
   assign {dmadr_imag, dmadr_real} = ram1_dmaing ? dr_ram1 : dr_ram0;


   // Per-RAM state transitions
   //
   // Multiple transitions can fire in the same cycle (e.g. a fill finishes
   // AND a compute slot opens up), but each ram only takes one state at end
   // of cycle thanks to the priority encoder structure below.

   always @(posedge clk) begin
      if (rst) begin
         ram0_state <= RAM_IDLE;
         ram1_state <= RAM_IDLE;
         fill_next  <= 1'b0;
      end else begin
         // ---------- ram 0 ----------
         case (ram0_state)
           RAM_IDLE:
              if (start_fill_ram0) ram0_state <= RAM_FILLING;
           RAM_FILLING:
              if (stream_done)    ram0_state <= RAM_READY_COMPUTE;
           RAM_READY_COMPUTE:
              if (!any_computing && autorun)
                                  ram0_state <= RAM_COMPUTING;
           RAM_COMPUTING:
              if (fin_fft)        ram0_state <= RAM_READY_DMA;
           RAM_READY_DMA:
              if (!any_dmaing)    ram0_state <= RAM_DMAING;
           RAM_DMAING:
              if (dma_done)       ram0_state <= RAM_IDLE;
           default:                ram0_state <= RAM_IDLE;
         endcase

         // ---------- ram 1 ----------
         case (ram1_state)
           RAM_IDLE:
              if (start_fill_ram1) ram1_state <= RAM_FILLING;
           RAM_FILLING:
              if (stream_done)    ram1_state <= RAM_READY_COMPUTE;
           RAM_READY_COMPUTE:
              // Only start compute on ram1 if ram0 isn't *also* about to
              // start (priority to ram0 to avoid both transitioning).
              if (!any_computing && autorun && !ram0_rcomp)
                                  ram1_state <= RAM_COMPUTING;
           RAM_COMPUTING:
              if (fin_fft)        ram1_state <= RAM_READY_DMA;
           RAM_READY_DMA:
              // Same priority: ram0_rdma wins.
              if (!any_dmaing && !ram0_rdma)
                                  ram1_state <= RAM_DMAING;
           RAM_DMAING:
              if (dma_done)       ram1_state <= RAM_IDLE;
           default:                ram1_state <= RAM_IDLE;
         endcase

         // Toggle fill_next when a fill is starting so the next new frame
         // goes to the other RAM.
         if (start_fill_ram0) fill_next <= 1'b1;
         if (start_fill_ram1) fill_next <= 1'b0;
      end
   end


   // Top-level status
   //   done       — at least one RAM has a result waiting for DMA
   //   status     — { dma_active, compute_active, filling_active }
   //   s_ready    — at least one RAM can absorb an input sample this cycle
   //                (i.e., already FILLING with room left, or IDLE so a new
   //                fill can start). Drops only when both RAMs are stuck
   //                downstream of FILLING with no fill in progress.

   assign done    = any_rdma || any_dmaing;
   assign status  = {any_dmaing, any_computing, any_filling};
   assign s_ready = any_filling || ram0_idle || ram1_idle;

endmodule // R2FFT