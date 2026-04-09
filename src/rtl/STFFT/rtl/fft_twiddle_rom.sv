module fft_twiddle_rom (
    input  wire       clk,
    input  wire       twact,
    input  wire [5:0] twa,
    output reg [15:0] twdr_cos
);
    always @(posedge clk) begin
        if (twact) begin
            `include "twiddle_rom_body.svh"
        end
    end
endmodule