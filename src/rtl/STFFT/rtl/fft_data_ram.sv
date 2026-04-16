// rtl/fft_data_ram.sv
// True dual-port RAM wrapper
// Simulation: behavioral dual-port
// ASIC: instantiates GF180 SRAMs with bank-switching (TODO)

module fft_data_ram (
    input  wire        clk,
    input  wire        ract,
    input  wire [6:0]  ra,      // independent read address
    output reg  [31:0] rdata,
    input  wire        wact,
    input  wire [6:0]  wa,      // independent write address
    input  wire [31:0] wdata
);
    // Behavioral dual-port memory for simulation
    // For ASIC synthesis this module will be replaced with
    // a bank-switched single-port SRAM implementation
    reg [31:0] mem [0:127];

    integer i;
    initial begin
        for (i = 0; i < 128; i = i + 1)
            mem[i] = 32'h0;
        rdata = 32'h0;
    end

    always @(posedge clk) begin
        if (wact)
            mem[wa] <= wdata;
        if (ract)
            rdata <= mem[ra];
    end

endmodule