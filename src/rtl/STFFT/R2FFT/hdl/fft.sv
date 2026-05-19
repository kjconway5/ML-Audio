`default_nettype none

// =============================================================================
// fft  —  user-facing wrapper around R2FFT with AXI-stream-style ready/valid
//
// Input port (s_axis-like):
//   i_valid + i_data  : producer asserts when a sample is offered
//   i_ready           : wrapper asserts when the FFT can take a sample now
//                       Transfer happens at the rising edge when both are high.
//
// Output port (m_axis-like):
//   o_valid + o_data  : wrapper asserts when an output sample is offered
//   o_ready           : consumer asserts when it can take a sample now
//                       Transfer happens at the rising edge when both are high.
//   o_last            : asserted with o_valid on the final sample of a frame.
//
// Notes:
//   - Input path is combinational pass-through from i_valid/i_data into R2FFT,
//     so i_ready reflects R2FFT.s_ready directly. No extra latency.
//   - Output path uses a 2-deep skid buffer (out + skid) so the DMA read
//     pipeline doesn't lose data when the consumer momentarily de-asserts
//     o_ready. Steady-state throughput is one sample per cycle.
//   - The wrapper auto-restarts DMA for each new frame's done assertion.
// =============================================================================
module fft #(
    parameter IW       = 16,
    parameter OW       = 16,
    parameter FFT_SIZE = 256,
    parameter FFT_N    = $clog2(FFT_SIZE)
)(
    input  wire                     i_clk,
    input  wire                     i_reset,

    // input stream (ready/valid)
    input  wire                     i_valid,
    input  wire signed [IW-1:0]     i_data,
    output wire                     i_ready,

    // output stream (ready/valid)
    output wire                     o_valid,
    output wire signed [2*OW-1:0]   o_data,
    input  wire                     o_ready,
    output wire                     o_last,

    // block-floating-point exponent for the currently-emitting frame
    output wire signed [7:0]        o_bfpexp
);

    // ---------------------------------------------------------------
    // Input front-end: combinational handshake into R2FFT
    //   sact_istream is the "transfer happens this cycle" signal.
    //   Imaginary half is tied to zero (real-valued input stream).
    // ---------------------------------------------------------------
    wire                  r2fft_s_ready;
    wire                  sact_istream = i_valid && r2fft_s_ready;
    wire signed [IW-1:0]  sdw_real     = i_data;
    wire signed [IW-1:0]  sdw_imag     = '0;

    assign i_ready = r2fft_s_ready;

    // ---------------------------------------------------------------
    // R2FFT instance + RAM/ROM wires
    // ---------------------------------------------------------------
    wire                  done;
    wire [2:0]            status;
    wire signed [7:0]     bfpexp;

    reg                   dmaact;
    reg  [FFT_N-1:0]      dmaa;
    wire signed [15:0]    dmadr_real, dmadr_imag;

    wire                  twact;
    wire [FFT_N-3:0]      twa;
    wire [15:0]           twdr_cos;

    wire                  act_ram0, we_ram0;
    wire [FFT_N-1:0]      a_ram0;
    wire [31:0]           dw_ram0, dr_ram0;

    wire                  act_ram1, we_ram1;
    wire [FFT_N-1:0]      a_ram1;
    wire [31:0]           dw_ram1, dr_ram1;

    R2FFT #(
        .FFT_LENGTH (FFT_SIZE),
        .FFT_DW     (16),
        .PL_DEPTH   (3)
    ) u_fft (
        .clk               (i_clk),
        .rst               (i_reset),

        .autorun           (1'b1),
        .run               (1'b0),
        .fin               (1'b0),
        .ifft              (1'b0),

        .done              (done),
        .status            (status),
        .bfpexp            (bfpexp),
        .s_ready           (r2fft_s_ready),

        .sact_istream      (sact_istream),
        .sdw_istream_real  (sdw_real),
        .sdw_istream_imag  (sdw_imag),

        .dmaact            (dmaact),
        .dmaa              (dmaa),
        .dmadr_real        (dmadr_real),
        .dmadr_imag        (dmadr_imag),

        .twact             (twact),
        .twa               (twa),
        .twdr_cos          (twdr_cos),

        // Single-port RAM 0
        .act_ram0          (act_ram0),
        .we_ram0           (we_ram0),
        .a_ram0            (a_ram0),
        .dw_ram0           (dw_ram0),
        .dr_ram0           (dr_ram0),

        // Single-port RAM 1
        .act_ram1          (act_ram1),
        .we_ram1           (we_ram1),
        .a_ram1            (a_ram1),
        .dw_ram1           (dw_ram1),
        .dr_ram1           (dr_ram1)
    );

    // Two single-port data RAMs (256 x 32)
    fft_data_ram u_ram0 (
        .clk (i_clk),
        .rst (i_reset),
        .act (act_ram0),
        .we  (we_ram0),
        .a   (a_ram0),
        .dw  (dw_ram0),
        .dr  (dr_ram0)
    );

    fft_data_ram u_ram1 (
        .clk (i_clk),
        .rst (i_reset),
        .act (act_ram1),
        .we  (we_ram1),
        .a   (a_ram1),
        .dw  (dw_ram1),
        .dr  (dr_ram1)
    );

    // Twiddle factor ROM — auto-generated 64-entry cosine LUT
    // (hardcoded for FFT_LENGTH=256: twa is 6 bits, FFT_N-2 = 6).
    fft_twiddle_rom u_twiddle_rom (
        .clk      (i_clk),
        .twact    (twact),
        .twa      (twa),
        .twdr_cos (twdr_cos)
    );

    // ---------------------------------------------------------------
    // Output DMA controller with 2-deep skid buffer
    //
    //   Pipeline timing (RAM has 1-cycle synchronous read):
    //     cycle K  : dma_advance=1, dmaa register holds the address being
    //                issued. R2FFT's RAM mux drives a_ram=dmaa, act=1.
    //     cycle K+1: dmaa_in_flight=1; dmadr now carries mem[issued addr].
    //                Capture into `out` if it's free, else into `skid`.
    //     cycle K+2: o_valid_r is high with o_data_r = the captured pair.
    //
    //   The skid slot absorbs a one-cycle backpressure stall without
    //   losing the in-flight read. dma_advance is gated on !skid_valid_n
    //   (the combinational next-state of skid), which means we only issue
    //   a new read if there will be room for its data when it lands.
    //   Steady-state throughput with a willing consumer is 1 sample/cycle.
    // ---------------------------------------------------------------
    reg                       dma_started;
    reg                       dmaa_in_flight;     // dmadr is fresh this cycle
    reg  [FFT_N-1:0]          in_flight_addr;     // addr whose data is on dmadr

    reg                       o_valid_r;
    reg  signed [2*OW-1:0]    o_data_r;
    reg                       o_last_r;

    reg                       skid_valid;
    reg  signed [2*OW-1:0]    skid_data;
    reg                       skid_last;

    wire                      transfer = o_valid_r && o_ready;
    wire signed [2*OW-1:0]    dmadr_pair     = {dmadr_real, dmadr_imag};
    wire                      in_flight_last = (in_flight_addr == FFT_SIZE-1);

    // Combinational next-state for out + skid.
    reg                       o_valid_n;
    reg  signed [2*OW-1:0]    o_data_n;
    reg                       o_last_n;
    reg                       skid_valid_n;
    reg  signed [2*OW-1:0]    skid_data_n;
    reg                       skid_last_n;

    always @(*) begin
        // Default: hold
        o_valid_n    = o_valid_r;
        o_data_n     = o_data_r;
        o_last_n     = o_last_r;
        skid_valid_n = skid_valid;
        skid_data_n  = skid_data;
        skid_last_n  = skid_last;

        // 1) Consumer takes the head — clear out
        if (transfer) begin
            o_valid_n = 1'b0;
            o_last_n  = 1'b0;   // bug fix: was held at o_last_r → stuck high between frames
        end

        // 2) If skid has data and out just freed (or was empty), move skid -> out
        if (skid_valid_n && !o_valid_n) begin
            o_data_n     = skid_data_n;
            o_last_n     = skid_last_n;
            o_valid_n    = 1'b1;
            skid_valid_n = 1'b0;
        end

        // 3) Fresh data arriving — land in out (if free) else in skid
        if (dmaa_in_flight) begin
            if (!o_valid_n) begin
                o_data_n  = dmadr_pair;
                o_last_n  = in_flight_last;
                o_valid_n = 1'b1;
            end else if (!skid_valid_n) begin
                skid_data_n  = dmadr_pair;
                skid_last_n  = in_flight_last;
                skid_valid_n = 1'b1;
            end
            // else: would-overflow — prevented by gating dma_advance below
        end
    end

    // Issue a new read only if the next-cycle skid slot is free. This
    // matches the requirement that whatever data arrives in 1 cycle must
    // have somewhere to land.
    wire dma_advance = dmaact && !skid_valid_n;

    always @(posedge i_clk) begin
        if (i_reset) begin
            dmaact         <= 1'b0;
            dmaa           <= '0;
            dma_started    <= 1'b0;
            dmaa_in_flight <= 1'b0;
            in_flight_addr <= '0;
            o_valid_r      <= 1'b0;
            o_data_r       <= '0;
            o_last_r       <= 1'b0;
            skid_valid     <= 1'b0;
            skid_data      <= '0;
            skid_last      <= 1'b0;
        end else begin
            // Buffer state advance
            o_valid_r  <= o_valid_n;
            o_data_r   <= o_data_n;
            o_last_r   <= o_last_n;
            skid_valid <= skid_valid_n;
            skid_data  <= skid_data_n;
            skid_last  <= skid_last_n;

            // 1-stage pipeline shift: dmadr is fresh the cycle after dma_advance
            dmaa_in_flight <= dma_advance;
            if (dma_advance) in_flight_addr <= dmaa;

            // DMA control + address counter
            if (done && !dma_started) begin
                dmaact      <= 1'b1;
                dmaa        <= '0;
                dma_started <= 1'b1;
            end else if (dma_advance) begin
                if (dmaa == FFT_SIZE-1)
                    dmaact <= 1'b0;        // last read issued; let pipeline drain
                else
                    dmaa <= dmaa + 1'b1;
            end

            // Clear dma_started once the entire pipeline has drained, so
            // the next frame's `done` re-arms a new DMA.
            if (dma_started && !dmaact && !dmaa_in_flight
                && !o_valid_n && !skid_valid_n) begin
                dma_started <= 1'b0;
            end
        end
    end

    assign o_valid = o_valid_r;
    assign o_data  = o_data_r;
    assign o_last  = o_last_r;
    assign o_bfpexp = bfpexp;

endmodule


// =============================================================================
// fft_data_ram  —  256 x 32-bit single-port (1RW) RAM
//
//   act = 1, we = 1  : write dw at address a
//   act = 1, we = 0  : read; dr valid 1 cycle later
//   act = 0          : idle (dr holds)
//
// SIM:    behavioural model with 1-cycle synchronous read latency.
// SYNTH:  4 GF180 single-port byte SRAMs in parallel, with the same
//         registered-Q pattern as your original. (NOTE: if your GF180
//         macro already has a registered Q, the extra latch makes this
//         a 2-cycle read — the R2FFT assumes 1 cycle. Drop the latch or
//         widen the butterfly schedule if you hit that.)
// =============================================================================
module fft_data_ram (
    input  wire        clk,
    input  wire        rst,

    input  wire        act,
    input  wire        we,
    input  wire [7:0]  a,
    input  wire [31:0] dw,
    output reg  [31:0] dr
);

`ifdef SIM

    reg [31:0] mem [0:255];

    always @(posedge clk) begin
        if (act) begin
            if (we) mem[a] <= dw;
            else    dr     <= mem[a];
        end
    end

`else

    // --------------------------------------------------------
    // GF180MCU single-port mapping (one access per cycle).
    //   CEN  = active-low chip enable               -> ~act
    //   GWEN = active-low global write enable       -> ~(act & we)
    //   WEN  = per-bit write mask (active low)      ->  8'h00 (all writable)
    // --------------------------------------------------------
    wire [7:0] q0, q1, q2, q3;

    gf180mcu_ocd_ip_sram__sram256x8m8wm1 u0 (
        .CLK  (clk),
        .CEN  (~act),
        .GWEN (~(act & we)),
        .WEN  (8'h00),
        .A    (a),
        .D    (dw[7:0]),
        .Q    (q0)
    );

    gf180mcu_ocd_ip_sram__sram256x8m8wm1 u1 (
        .CLK  (clk),
        .CEN  (~act),
        .GWEN (~(act & we)),
        .WEN  (8'h00),
        .A    (a),
        .D    (dw[15:8]),
        .Q    (q1)
    );

    gf180mcu_ocd_ip_sram__sram256x8m8wm1 u2 (
        .CLK  (clk),
        .CEN  (~act),
        .GWEN (~(act & we)),
        .WEN  (8'h00),
        .A    (a),
        .D    (dw[23:16]),
        .Q    (q2)
    );

    gf180mcu_ocd_ip_sram__sram256x8m8wm1 u3 (
        .CLK  (clk),
        .CEN  (~act),
        .GWEN (~(act & we)),
        .WEN  (8'h00),
        .A    (a),
        .D    (dw[31:24]),
        .Q    (q3)
    );

    // Registered read collation. Only updates on read cycles so writes
    // don't clobber dr with spurious bus state.
    always @(posedge clk) begin
        if (act && !we)
            dr <= {q3, q2, q1, q0};
    end

`endif

endmodule

`default_nettype wire