// full_demo_top.sv — Stage 3 end-to-end inference top for Arty A7-100t.
//
// Builds on Stage 2 by adding the DSCNN. The data path is:
//   host UART ─► boot_controller ─► features_router  ─► pipeline_top.flash_*
//                              ─► dscnn_router       ─► kws_top.{w_*, cfg_*}
//                              ─► audio_boot         ─► pipeline_top.{data_i, valid_i}
//                              ─► (ACK / NACK)
//   pipeline_top.sp_{a,b}_*    ─► kws_top.sp_{a,b}_*   (real spectrogram SRAM)
//                              ─► spect_capture        (debug tap; same writes)
//   kws_top.done, class_out    ─► class_reporter       ─► tagged byte over UART
//   host DBG_READ_SPECT_{A,B}  ─► spect_streamer       (still wired for debug)
//
// CIC + comp-FIR remain bypassed (16 kHz audio in via MOD_AUDIO).
// `kws_top.start` is tied to boot_done — once boot completes, every
// spect_done pulse from the features pipeline fires one inference and
// the result is emitted automatically.
//
// Status LEDs:
//   LD4 — boot_done
//   LD5 — inference counter LSB (toggles per inference done)
//   LD6 — RX byte counter LSB
//   LD7 — sticky NACK / framing / overrun error (clear with BTN0)

module full_demo_top
    import boot_pkg::*;
#(
    parameter logic [15:0] UART_PRESCALE = 16'd27,
    parameter int          ADDR_W        = 11
) (
    input  wire CLK100MHZ,

    input  wire btn0,

    input  wire uart_txd_in,
    output wire uart_rxd_out,

    // Class-indicator LEDs, named to match the Arty A7 silkscreen.
    // led4..led7 are the 4 green monochrome LEDs along the edge.
    // led1..led3 are the green channel of 3 of the 4 RGB LEDs, PWM-
    // dimmed because the RGB LEDs are noticeably brighter than the
    // monochrome ones at full-on. LD0 is intentionally unused.
    //   led7 yes     led3 wow
    //   led6 on      led2 silence
    //   led5 off     led1 unknown
    //   led4 no
    // Class is latched on kws_done so the LED shows the most recent
    // classification until the next inference fires.
    output wire led1,
    output wire led2,
    output wire led3,
    output wire led4,
    output wire led5,
    output wire led6,
    output wire led7
);


    //  Reset synchronizer
    logic rst_meta, rst_sync;
    always_ff @(posedge CLK100MHZ) begin
        rst_meta <= btn0;
        rst_sync <= rst_meta;
    end

    // Soft session reset — pulsed by boot_controller on
    // CTRL_SESSION_RESET. Stretched to 8 cycles to ensure every sync
    // reset downstream sees it. OR'd with the main reset to feed
    // pipeline_top + kws_top + spect_capture; the SRAMs inside those
    // modules use behavioral cells with no reset clause, so their
    // contents survive.
    logic       session_reset_pulse;
    logic [2:0] sess_rst_extend;
    logic       sess_rst_active;
    always_ff @(posedge CLK100MHZ) begin
        if (rst_sync) begin
            sess_rst_extend <= 3'b0;
        end else if (session_reset_pulse) begin
            sess_rst_extend <= 3'b111;
        end else if (sess_rst_extend != 0) begin
            sess_rst_extend <= sess_rst_extend - 3'd1;
        end
    end
    assign sess_rst_active = (sess_rst_extend != 0);

    wire data_path_reset = rst_sync | sess_rst_active;


    //  TX mux
    //
    //  Priority (high → low):
    //    1. boot_controller  — ACK/NACK/ERR must go first; this includes
    //       the ACK that triggers spect_streamer in the first place.
    //    2. spect_streamer    — once busy_o asserts (after the ACK), it
    //       OWNS the TX line until the last byte. This locks out
    //       class_reporter so class-tag bytes can't interleave into the
    //       spectrogram data stream (previously corrupted --read-spect).
    //    3. class_reporter    — normal inference-result emission when
    //       no debug read is in flight.
    //
    //  Note: spect_streamer.tx_valid is 0 during ST_SETUP/ST_WAIT (SRAM
    //  read latency); the gap is held by ss_busy=1 so cr stays masked.
    logic [7:0] rx_byte;
    logic       rx_valid, rx_ready;

    logic [7:0] bc_tx_byte, cr_tx_byte, ss_tx_byte;
    logic       bc_tx_valid, cr_tx_valid, ss_tx_valid;
    logic       bc_tx_ready, cr_tx_ready, ss_tx_ready;
    logic       ss_busy;   // declared & driven below at the spect_streamer inst

    logic [7:0] tx_byte;
    logic       tx_valid, tx_ready;

    always_comb begin
        if (bc_tx_valid) begin
            tx_byte     = bc_tx_byte;
            tx_valid    = 1'b1;
        end else if (ss_busy) begin
            // Spect_streamer owns the line: forward its byte (or nothing
            // during SRAM latency cycles), do NOT fall through to cr.
            tx_byte     = ss_tx_byte;
            tx_valid    = ss_tx_valid;
        end else if (cr_tx_valid) begin
            tx_byte     = cr_tx_byte;
            tx_valid    = 1'b1;
        end else begin
            tx_byte     = 8'h00;
            tx_valid    = 1'b0;
        end
    end

    assign bc_tx_ready = bc_tx_valid & tx_ready;
    assign ss_tx_ready = ~bc_tx_valid & ss_busy & tx_ready;
    assign cr_tx_ready = ~bc_tx_valid & ~ss_busy & cr_tx_valid & tx_ready;


    //  UART
    logic rx_overrun_err, rx_frame_err;

    uart #(.DATA_WIDTH(8)) u_uart (
        .clk              (CLK100MHZ),
        .rst              (rst_sync),

        .s_axis_tdata     (tx_byte),
        .s_axis_tvalid    (tx_valid),
        .s_axis_tready    (tx_ready),

        .m_axis_tdata     (rx_byte),
        .m_axis_tvalid    (rx_valid),
        .m_axis_tready    (rx_ready),

        .rxd              (uart_txd_in),
        .txd              (uart_rxd_out),

        .tx_busy          (),
        .rx_busy          (),
        .rx_overrun_error (rx_overrun_err),
        .rx_frame_error   (rx_frame_err),

        .prescale         (UART_PRESCALE)
    );


    //  Boot controller + routers
    boot_bus_t features_boot, dscnn_boot, audio_boot;

    logic        boot_done;
    logic        pkt_valid;
    logic [7:0]  last_target;
    logic [15:0] last_addr, last_len;

    boot_controller u_boot_ctrl (
        .clk_i           (CLK100MHZ),
        .reset_i         (rst_sync),

        .rx_byte_i       (rx_byte),
        .rx_valid_i      (rx_valid),
        .rx_ready_o      (rx_ready),

        .tx_byte_o       (bc_tx_byte),
        .tx_valid_o      (bc_tx_valid),
        .tx_ready_i      (bc_tx_ready),

        .features_boot_o (features_boot),
        .dscnn_boot_o    (dscnn_boot),
        .audio_boot_o    (audio_boot),

        .boot_done_o     (boot_done),
        .pkt_valid_o     (pkt_valid),
        .session_reset_o (session_reset_pulse),
        .last_target_o   (last_target),
        .last_addr_o     (last_addr),
        .last_len_o      (last_len)
    );

    logic        lut_we;
    logic [5:0]  lut_addr;
    logic [15:0] lut_data;
    logic        mel_we;
    logic [7:0]  mel_addr;
    logic [15:0] mel_data;
    logic        meta_we;
    logic [7:0]  meta_addr;
    logic [7:0]  meta_data;
    // VAD threshold register (latches FEAT_VAD_THRESH packets; resets
    // to 0 ⇒ VAD bypassed). Persists across session_reset, same as the
    // SRAMs, because the router lives in the boot domain (rst_sync).
    logic [31:0] vad_threshold;
    // Per-checkpoint input requant multiplier (FEAT_INPUT_QUANT_MULT).
    // Reset to 0 means "use the RTL parameter default (5817845)" so a
    // demo boot that doesn't program it stays compatible with the
    // hardcoded value. Each model's input_quant.txt has its own value.
    logic [31:0] input_quant_mult;

    features_boot_router u_feat_router (
        .clk_i               (CLK100MHZ),
        .reset_i             (rst_sync),
        .boot_i              (features_boot),
        .lut_boot_we_o       (lut_we),
        .lut_boot_addr_o     (lut_addr),
        .lut_boot_wdata_o    (lut_data),
        .mel_boot_we_o       (mel_we),
        .mel_boot_addr_o     (mel_addr),
        .mel_boot_wdata_o    (mel_data),
        .meta_boot_we_o      (meta_we),
        .meta_boot_addr_o    (meta_addr),
        .meta_boot_wdata_o   (meta_data),
        .vad_threshold_o     (vad_threshold),
        .input_quant_mult_o  (input_quant_mult)
    );

    logic        w_we;
    logic [12:0] w_waddr;
    logic [7:0]  w_wdata;
    logic        b_we;
    logic [10:0] b_waddr;
    logic [7:0]  b_wdata;
    logic        cfg_we;
    logic [7:0]  cfg_addr;
    logic [7:0]  cfg_wdata;

    dscnn_boot_router u_dscnn_router (
        .boot_i           (dscnn_boot),
        .w_boot_we_o      (w_we),
        .w_boot_addr_o    (w_waddr),
        .w_boot_wdata_o   (w_wdata),
        .b_boot_we_o      (b_we),
        .b_boot_addr_o    (b_waddr),
        .b_boot_wdata_o   (b_wdata),
        .cfg_boot_we_o    (cfg_we),
        .cfg_boot_addr_o  (cfg_addr),
        .cfg_boot_wdata_o (cfg_wdata)
    );


    //  Audio routing (MOD_AUDIO 16-bit packed → pipeline_top sample stream)
    logic signed [15:0] audio_sample;
    logic               audio_valid;
    assign audio_sample = $signed(audio_boot.data);
    assign audio_valid  = audio_boot.valid;


    //  Features pipeline
    logic               sp_a_we;
    logic [ADDR_W-1:0]  sp_a_waddr;
    logic signed [7:0]  sp_a_wdata;
    logic               sp_b_we;
    logic [ADDR_W-1:0]  sp_b_waddr;
    logic signed [7:0]  sp_b_wdata;

    logic               spect_done;
    logic               spect_write_sel;

    pipeline_top u_pipe (
        .clk_i                  (CLK100MHZ),
        .reset_i                (data_path_reset),

        .data_i                 (audio_sample),
        .valid_i                (audio_valid),

        .sp_a_we                (sp_a_we),
        .sp_a_waddr             (sp_a_waddr),
        .sp_a_wdata             (sp_a_wdata),

        .sp_b_we                (sp_b_we),
        .sp_b_waddr             (sp_b_waddr),
        .sp_b_wdata             (sp_b_wdata),

        .spect_done             (spect_done),
        .spect_write_sel        (spect_write_sel),

        .mel_compensated_o      (),
        .mel_compensated_valid_o(),

        // VAD threshold from features_boot_router (FEAT_VAD_THRESH).
        // Defaults to 0 ⇒ VAD bypassed if the host never sends one.
        .vad_threshold_i        (vad_threshold),
        .vad_active_o           (),  // not exposed on the demo (could be LED)
        .dft_vad_obs_en_i       (1'b0),
        .vad_frame_drop_ol      (),  // not exposed on the demo

        // Per-checkpoint input quant multiplier from features_boot_router
        // (FEAT_INPUT_QUANT_MULT). 0 ⇒ pipeline_top falls back to its
        // compile-time parameter default (5817845).
        .input_quant_mult_i     (input_quant_mult),

        .flash_mel_coeff_we_i   (mel_we),
        .flash_mel_coeff_addr_i (mel_addr),
        .flash_mel_coeff_data_i (mel_data),

        .flash_mel_index_we_i   (meta_we),
        .flash_mel_index_addr_i (meta_addr),
        .flash_mel_index_data_i (meta_data),

        .flash_log_lut_we_i     (lut_we),
        .flash_log_lut_addr_i   (lut_addr),
        .flash_log_lut_data_i   (lut_data)
        // test_* ports removed: the post-merge streaming-STFFT rework
        // (b59d524) rewrote pipeline_top and dropped test-mode entirely.
        // These four were tying test mode OFF, so removing them is
        // behavior-preserving.
    );


    //  Spect capture (debug tap, same writes as kws_top)
    logic [ADDR_W-1:0] spect_raddr;
    logic              spect_bank_sel;
    logic signed [7:0] spect_rdata;
    logic [7:0]        spect_done_count;
    logic              last_write_sel;

    spect_capture #(.ADDR_W(ADDR_W)) u_spect_cap (
        .clk              (CLK100MHZ),
        .reset            (data_path_reset),
        .sp_a_we          (sp_a_we),
        .sp_a_waddr       (sp_a_waddr),
        .sp_a_wdata       (sp_a_wdata),
        .sp_b_we          (sp_b_we),
        .sp_b_waddr       (sp_b_waddr),
        .sp_b_wdata       (sp_b_wdata),
        .spect_done       (spect_done),
        .spect_write_sel  (spect_write_sel),
        .read_bank_sel    (spect_bank_sel),
        .raddr            (spect_raddr),
        .rdata            (spect_rdata),
        .spect_done_count (spect_done_count),
        .last_write_sel   (last_write_sel)
    );

    // ss_busy declared in the TX-mux block above (used by the mux to
    // lock out class_reporter for the duration of a debug-read burst).
    spect_streamer #(.ADDR_W(ADDR_W)) u_spect_stream (
        .clk_i             (CLK100MHZ),
        .reset_i           (rst_sync),
        .pkt_valid_i       (pkt_valid),
        .last_target_i     (last_target),
        .last_addr_i       (last_addr),
        .last_len_i        (last_len),
        .spect_raddr_o     (spect_raddr),
        .spect_bank_sel_o  (spect_bank_sel),
        .spect_rdata_i     (spect_rdata),
        .tx_byte_o         (ss_tx_byte),
        .tx_valid_o        (ss_tx_valid),
        .tx_ready_i        (ss_tx_ready),
        .busy_o            (ss_busy)
    );


    //  KWS / DS-CNN
    //
    // start is held high after boot_done; FSM's own (start & cfg_load_done
    // & spect_ready & weights_ready) gate ensures it only fires on a new
    // spectrogram. cfg_load_done is asserted by writing cfg_addr=0xFF
    // (host's cfg.hex includes the sentinel as its last byte).

    logic       kws_done;
    logic [2:0] kws_class;

    // NOTE: kws_top is NOT pulled into the session_reset domain — its
    // `reset` would clear cfg_load_done, forcing the host to re-send
    // the cfg sentinel each cycle. With FSM.sv's spect_ready consume
    // fix in place, the FSM idles cleanly in IDLE between inferences,
    // so a soft reset isn't required here. Weight/cfg/spect SRAMs
    // already survive any reset (behavioral, no reset clause).
    kws_top u_kws (
        .clk             (CLK100MHZ),
        .reset           (rst_sync),
        .start           (boot_done),
        .done            (kws_done),
        .class_out       (kws_class),

        .cfg_we          (cfg_we),
        .cfg_addr        (cfg_addr),
        .cfg_wdata       (cfg_wdata),

        .w_we            (w_we),
        .w_waddr         (w_waddr),
        .w_wdata         (w_wdata),

        .b_we            (b_we),
        .b_waddr         (b_waddr),
        .b_wdata         (b_wdata),

        .spect_done      (spect_done),
        .spect_write_sel (spect_write_sel),

        .sp_a_we         (sp_a_we),
        .sp_a_waddr      (sp_a_waddr),
        .sp_a_wdata      (sp_a_wdata),

        .sp_b_we         (sp_b_we),
        .sp_b_waddr      (sp_b_waddr),
        .sp_b_wdata      (sp_b_wdata),

        // DFT observability disabled for the demo (debug taps are
        // outputs, left unconnected — synth will trim them).
        .test_mode_ml    (1'b0)
    );


    //  Class reporter — pushes 0xC0|class on every kws_done rising edge.
    logic cr_busy;
    class_reporter u_class_rep (
        .clk_i      (CLK100MHZ),
        .reset_i    (rst_sync),
        .done_i     (kws_done),
        .class_i    (kws_class),
        .tx_byte_o  (cr_tx_byte),
        .tx_valid_o (cr_tx_valid),
        .tx_ready_i (cr_tx_ready),
        .busy_o     (cr_busy)
    );


    //  Class-indicator LEDs
    //
    // Class enum (sorted alphabetical, matches the trained checkpoint
    // and class_reporter): 0=no, 1=off, 2=on, 3=silence, 4=unknown,
    // 5=wow, 6=yes. `last_class` latches kws_class on every kws_done
    // so the LEDs hold "the most recent classification" between
    // inference firings. Reset value is silence so led2 is on at
    // power-up — a clear "I am up and quiet" indicator.
    logic [2:0] last_class;
    always_ff @(posedge CLK100MHZ) begin
        if (rst_sync)        last_class <= 3'd3;        // silence
        else if (kws_done)   last_class <= kws_class;
    end

    // Monochrome LEDs: full-on when the class matches.
    assign led7 = (last_class == 3'd6);   // yes
    assign led6 = (last_class == 3'd2);   // on
    assign led5 = (last_class == 3'd1);   // off
    assign led4 = (last_class == 3'd0);   // no

    // PWM dimmer for the RGB green channels. The RGB LEDs on this
    // board are visibly brighter than the monochrome ones at full-on;
    // a low-duty PWM keeps them roughly matched. 8-bit counter at
    // 100 MHz ⇒ ~390 kHz PWM (well above flicker), 16/256 ≈ 6% duty.
    // Bump LED_DIM_THRESH if 6% is too dim for your room.
    localparam logic [7:0] LED_DIM_THRESH = 8'd16;
    logic [7:0] pwm_cnt;
    always_ff @(posedge CLK100MHZ) begin
        if (rst_sync) pwm_cnt <= 8'd0;
        else          pwm_cnt <= pwm_cnt + 8'd1;
    end
    wire dim_en = (pwm_cnt < LED_DIM_THRESH);

    assign led3 = dim_en & (last_class == 3'd5);   // wow
    assign led2 = dim_en & (last_class == 3'd3);   // silence
    assign led1 = dim_en & (last_class == 3'd4);   // unknown

endmodule
