// Behavioral simulation model for GF180MCU 128x8 SRAM macro
// For simulation ONLY — not for synthesis
// Matches port interface from SPICE subckt:
//   CLK, CEN (active-low chip enable), GWEN (active-low global write enable),
//   WEN[7:0] (active-low per-bit write enable), A[6:0], D[7:0], Q[7:0]

module gf180mcu_fd_ip_sram__sram128x8m8wm1 (
    input  wire       CLK,
    input  wire       CEN,     // active low: 0 = enabled
    input  wire       GWEN,    // active low: 0 = write, 1 = read
    input  wire [7:0] WEN,     // active low per-bit: 0 = write that bit
    input  wire [6:0] A,
    input  wire [7:0] D,
    output reg  [7:0] Q
);
    reg [7:0] mem [0:127];

    // Initialize to zero to avoid X propagation in sim
    integer j;
    initial begin
        for (j = 0; j < 128; j = j + 1)
            mem[j] = 8'h00;
        Q = 8'h00;
    end

    always @(posedge CLK) begin
        if (!CEN) begin          // chip enabled
            if (!GWEN) begin     // write mode
                // per-bit write: WEN=0 means write that bit
                mem[A] <= (D & ~WEN) | (mem[A] & WEN);
            end else begin       // read mode
                Q <= mem[A];
            end
        end
    end

endmodule
