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