/*
Calculate Power/Magnitude = real^2 + imag^2 from STFT output
*/
import lau_pkg::*;

module power_calc #(
    parameter int IW = 18, // input width from STFT
    parameter int SHIFT = 6
)(
    input  logic [IW-1:0]       real_il, // fft_data[35:18]
    input  logic [IW-1:0]       imag_il, // fft_data[17:0]
    input  logic                valid_il,
    output logic [(2*IW)-SHIFT+1:0] power_ol,
    output logic                valid_ol
);

    logic [2*IW-1:0] real_sq, imag_sq;
    logic [2*IW:0]   sum_full;

    // Instantiate two squarers — speed set here as parameter
    SqrSgn #(
        .width(IW),
        .speed(lau_pkg::FAST)   // <-- this is where you choose
    ) u_re_sq (
        .X(real_il),
        .P(real_sq)
    );

    SqrSgn #(
        .width(IW),
        .speed(lau_pkg::FAST)   // <-- same here
    ) u_im_sq (
        .X(imag_il),
        .P(imag_sq)
    );

    // Sum and scale
    assign sum_full  = {1'b0, real_sq} + {1'b0, imag_sq};  // 37-bit
    assign power_out = sum_full[2*IW:SHIFT];            // drop bottom SHIFT bits

    always_ff @(posedge clk) begin
        valid_out <= valid_in;
    end

endmodule
