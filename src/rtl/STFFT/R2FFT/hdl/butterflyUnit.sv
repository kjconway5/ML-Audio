/* ===========================================================================
 * butterflyUnit
 *
 * State-machine memory sequencer wrapped around butterflyCore (unchanged).
 *
 * Per butterfly (7-state walk):
 *
 *     S_READ_A        : drive RAM read at addrA
 *     S_READ_B        : drive RAM read at addrB.   dr = x[a]; latch.
 *     S_LAUNCH        : iact pulse to butterflyCore. dr = x[b]; feed live.
 *     S_WAIT_COMPUTE  : RAM idle, waiting for butterflyCore.oact (L cycles).
 *     S_WRITE_A       : drive RAM write y[a] at addrA_lat
 *     S_WRITE_B       : drive RAM write y[b] at addrB_lat. advance bf_cnt.
 *
 * Cycles per butterfly = 5 + L where L = PL_DEPTH + 2 (butterflyCore is
 * two ramPipelineBridge instances around radix2Butterfly, each adding 1
 * cycle). For PL_DEPTH=3 that's 10 cycles/BF. Total FFT compute for
 * N=256: 10 * 128 * 8 = 10240 cycles.
 *
 * This is the safe-and-simple flavor (no port collisions for any
 * PL_DEPTH). The optimal scheme (4 cycles/BF with a write queue of
 * depth ceil(L/4)) is left as a follow-up.
 *
 * butterflyCore.ictrl is held at 2'b10 — the bridges' pass-through
 * mode. With ictrl=2'b10 the bridge muxes pick (ev0Data, od0Data),
 * which are the current-cycle inputs; this is what we want since x[a]
 * and x[b] are already correctly paired at the input here. (The
 * original generator drove ictrl=2'b00 / 2'b11 alternately at stage>=1
 * to recover pairs that the dual-RAM layout split across cycles — we
 * don't have that split anymore.)
 *
 * Address scheme per stage s, butterfly index bf in 0..N/2-1:
 *     addrA = { bf[FFT_N-2:s], 1'b0, bf[s-1:0] }
 *     addrB = { bf[FFT_N-2:s], 1'b1, bf[s-1:0] }
 * (i.e. insert a 0/1 at bit position s)
 * ===========================================================================
 */
module butterflyUnit
  #(parameter FFT_N         = 10,
    parameter FFT_DW        = 16,
    parameter FFT_BFPDW     = 5,
    parameter PL_DEPTH      = 0,
    parameter STAGE_COUNT_BW = 4)
   (
    input  wire                       clk,
    input  wire                       rst,
 
    input  wire                       clr_bfp,
    input  wire [FFT_BFPDW-1:0]       ibfp,
    output wire [FFT_BFPDW-1:0]       obfp,
 
    input  wire                       run,
    input  wire [STAGE_COUNT_BW-1:0]  stageCount,
    output wire                       iteratorDone,
    output wire                       oact,
    input  wire                       ifft,
 
    // twiddle ROM
    output wire                       twact,
    output wire [FFT_N-1-2:0]         twa,
    input  wire [FFT_DW-1:0]          twdr_cos,
 
    // single-port RAM
    output reg                        act,
    output reg                        we,
    output reg [FFT_N-1:0]            a,
    output reg [FFT_DW*2-1:0]         dw,
    input  wire [FFT_DW*2-1:0]        dr
    );
 

   // Sequencer states

   typedef enum logic [2:0] {
      S_IDLE          = 3'd0,
      S_READ_A        = 3'd1,
      S_READ_B        = 3'd2,
      S_LAUNCH        = 3'd3,
      S_WAIT_COMPUTE  = 3'd4,
      S_WRITE_A       = 3'd5,
      S_WRITE_B       = 3'd6,
      S_DONE          = 3'd7
   } seq_t;
 
   seq_t state_f, state_n;
   reg [FFT_N-1-1:0] bf_cnt;
   wire bf_cnt_full = (&bf_cnt);
 

   // Address generation for current butterfly at current stage
   //   addrA[i] = bf_cnt[i]      if i <  stageCount
   //              0              if i == stageCount
   //              bf_cnt[i-1]    if i >  stageCount
   //   addrB same with bit stageCount = 1

   reg [FFT_N-1:0] addrA, addrB;
   integer i;
   always_comb begin
      addrA = '0;
      addrB = '0;
      for (i = 0; i < FFT_N; i = i + 1) begin
         if (i < stageCount) begin
            addrA[i] = bf_cnt[i];
            addrB[i] = bf_cnt[i];
         end else if (i == stageCount) begin
            addrA[i] = 1'b0;
            addrB[i] = 1'b1;
         end else begin
            addrA[i] = bf_cnt[i-1];
            addrB[i] = bf_cnt[i-1];
         end
      end
   end
 

   // Twiddle factor address = bit_reverse( bf_cnt >> stageCount )

   wire [FFT_N-1-1:0] tw_idx = bf_cnt >> stageCount;
   wire [FFT_N-1-1:0] tw_idx_rev;
   genvar gi;
   generate
      for (gi = 0; gi <= FFT_N-1-1; gi = gi + 1) begin : TW_REV
         assign tw_idx_rev[gi] = tw_idx[FFT_N-1-1-gi];
      end
   endgenerate
 
   wire [FFT_DW:0] tdr_rom_real, tdr_rom_imag;
   // Twiddle launch is asserted one cycle BEFORE bfcore_iact. Reason:
   //   - ramPipelineBridge has 2 register stages (actPipe -> oact_f), so
   //     butterflyCore.iact at cycle T arrives at radix2Butterfly.iact at
   //     T+2; that's the cycle radix2Butterfly samples twiddle_real/imag.
   //   - twiddleFactorRomBridge takes 3 cycles from tact_rom to tdr_rom_*
   //     (cos request -> sin request -> latch cosReadData_2 / sinReadData_2).
   //   - So we need tact_rom at T-1 for the twiddle to be valid at T+2.
   // bf_cnt and stageCount are stable across S_READ_A/B/LAUNCH, so the
   // twiddle address is correct in S_READ_B.
   twiddleFactorRomBridge
     #(.FFT_N(FFT_N), .FFT_DW(FFT_DW))
   utwiddleFactorRomBridge
     (.clk           (clk),
      .rst           (rst),
      .tact_rom      (state_f == S_READ_B),
      .evenOdd       (1'b0),
      .ifft          (ifft),
      .ta_rom        (tw_idx_rev),
      .tdr_rom_real  (tdr_rom_real),
      .tdr_rom_imag  (tdr_rom_imag),
      .twact         (twact),
      .twa           (twa),
      .twdr_cos      (twdr_cos));
 

   // Per-butterfly latches: x[a], the two addresses, the two results

   reg [FFT_DW*2-1:0]   xa_lat;
   reg [FFT_N-1:0]      addrA_lat, addrB_lat;
   reg [FFT_DW*2-1:0]   yEven_lat, yOdd_lat;
 

   // butterflyCore instance

   wire                   bfcore_oact;
   wire [1:0]             bfcore_octrl;
   wire [FFT_N-1-1:0]     bfcore_oMemAddr;
   wire [FFT_DW*2-1:0]    bfcore_oEven;
   wire [FFT_DW*2-1:0]    bfcore_oOdd;
   wire [FFT_BFPDW-1:0]   bw_ramwrite;
 
   wire bfcore_iact = (state_f == S_LAUNCH);
 
   butterflyCore
     #(.FFT_N    (FFT_N),
       .FFT_DW   (FFT_DW),
       .FFT_BFPDW(FFT_BFPDW),
       .PL_DEPTH (PL_DEPTH))
   ubutterflyCore
     (.clk          (clk),
      .rst          (rst),
      .clr_bfp      (clr_bfp),
      .bw_ramwrite  (bw_ramwrite),
      .ibfp         (ibfp),
      .iact         (bfcore_iact),
      .ictrl        (2'b10),                 // bridges in pass-through mode
      .oact         (bfcore_oact),
      .octrl        (bfcore_octrl),
      .iMemAddr     ('0),                    // unused; addresses kept locally
      .iEvenData    (xa_lat),                // x[a] (latched in S_READ_B)
      .iOddData     (dr),                    // x[b] live on dr at S_LAUNCH
      .oMemAddr     (bfcore_oMemAddr),
      .oEvenData    (bfcore_oEven),
      .oOddData     (bfcore_oOdd),
      .twiddle_real (tdr_rom_real),
      .twiddle_imag (tdr_rom_imag));
 

   // State machine

   always_comb begin
      state_n = state_f;
      case (state_f)
        S_IDLE:          state_n = S_READ_A;
        S_READ_A:        state_n = S_READ_B;
        S_READ_B:        state_n = S_LAUNCH;
        S_LAUNCH:        state_n = S_WAIT_COMPUTE;
        S_WAIT_COMPUTE:  state_n = bfcore_oact ? S_WRITE_A : S_WAIT_COMPUTE;
        S_WRITE_A:       state_n = S_WRITE_B;
        S_WRITE_B:       state_n = bf_cnt_full ? S_DONE : S_READ_A;
        S_DONE:          state_n = S_DONE;
        default:         state_n = S_IDLE;
      endcase
   end
 
   always @(posedge clk) begin
      if (rst || !run) state_f <= S_IDLE;
      else             state_f <= state_n;
   end
 
   // bf_cnt advances at the end of each butterfly (S_WRITE_B)
   always @(posedge clk) begin
      if (rst || !run) bf_cnt <= '0;
      else if ((state_f == S_WRITE_B) && !bf_cnt_full)
         bf_cnt <= bf_cnt + 1'b1;
   end
 
   // Latch x[a] and addresses during the read phases
   always @(posedge clk) begin
      if (state_f == S_READ_B) xa_lat <= dr;
      if (state_f == S_READ_A) addrA_lat <= addrA;
      if (state_f == S_READ_B) addrB_lat <= addrB;
 
      // Capture compute results when butterflyCore signals done
      if (bfcore_oact) begin
         yEven_lat <= bfcore_oEven;
         yOdd_lat  <= bfcore_oOdd;
      end
   end
 

   // Drive RAM port per state

   always_comb begin
      act = 1'b0;
      we  = 1'b0;
      a   = '0;
      dw  = '0;
      case (state_f)
        S_READ_A:  begin act = 1'b1; we = 1'b0; a = addrA;     end
        S_READ_B:  begin act = 1'b1; we = 1'b0; a = addrB;     end
        S_WRITE_A: begin act = 1'b1; we = 1'b1; a = addrA_lat; dw = yEven_lat; end
        S_WRITE_B: begin act = 1'b1; we = 1'b1; a = addrB_lat; dw = yOdd_lat;  end
        default: ;
      endcase
   end
 

   // Status outputs to sub-sequencer
   //   oact = high during the 2-cycle writeback window of each BF; clears
   //          once we reach S_DONE so the sub-sequencer can advance stage.
   //   iteratorDone = high once we've completed the last BF of this stage.

   assign oact         = (state_f == S_WRITE_A) || (state_f == S_WRITE_B);
   assign iteratorDone = (state_f == S_DONE);
 

   // BFP max tracker on write-back data
   bfp_maxBitWidth #(.FFT_BFPDW(FFT_BFPDW)) ubfp_maxBitWidth
     (.clk    (clk),
      .rst    (rst),
      .clr    (clr_bfp),
      .bw_act (oact),
      .bw     (bw_ramwrite),
      .max_bw (obfp));
 
endmodule // butterflyUnit
 
