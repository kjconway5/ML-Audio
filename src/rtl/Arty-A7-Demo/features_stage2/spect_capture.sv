// spect_capture.sv — Stage 2: sniff sp_a/sp_b writes from pipeline_top
// into a local 2-bank behavioral SRAM so the host can DBG_READ the
// finished spectrogram back over UART.
//
// In the real chip, the spectrogram lives inside `kws_top.spectrogram_sram`
// and is consumed directly by the DSCNN. For Stage 2 (features-only)
// we want to read it out and compare against the golden model.

module spect_capture #(
    parameter int ADDR_W = 11
) (
    input  wire                  clk,
    input  wire                  reset,

    // Writes from pipeline_top
    input  wire                  sp_a_we,
    input  wire [ADDR_W-1:0]     sp_a_waddr,
    input  wire signed [7:0]     sp_a_wdata,

    input  wire                  sp_b_we,
    input  wire [ADDR_W-1:0]     sp_b_waddr,
    input  wire signed [7:0]     sp_b_wdata,

    input  wire                  spect_done,
    input  wire                  spect_write_sel,

    // Verify read (one shared port, bank selected externally)
    input  wire                  read_bank_sel,   // 0 = A, 1 = B
    input  wire [ADDR_W-1:0]     raddr,
    output logic signed [7:0]    rdata,

    // Status / debug
    output logic [7:0]           spect_done_count,
    output logic                 last_write_sel
);

    logic signed [7:0] mem_a [0:(1<<ADDR_W)-1];
    logic signed [7:0] mem_b [0:(1<<ADDR_W)-1];

    always_ff @(posedge clk) begin
        if (sp_a_we) mem_a[sp_a_waddr] <= sp_a_wdata;
        if (sp_b_we) mem_b[sp_b_waddr] <= sp_b_wdata;
    end

    logic signed [7:0] rdata_q;
    always_ff @(posedge clk) begin
        rdata_q <= read_bank_sel ? mem_b[raddr] : mem_a[raddr];
    end
    assign rdata = rdata_q;

    always_ff @(posedge clk) begin
        if (reset) begin
            spect_done_count <= 8'd0;
            last_write_sel   <= 1'b0;
        end else if (spect_done) begin
            spect_done_count <= spect_done_count + 8'd1;
            last_write_sel   <= spect_write_sel;
        end
    end

endmodule
