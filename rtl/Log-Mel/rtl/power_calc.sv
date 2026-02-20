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

    // squarers for real and imaginary components
    SqrSgn #(
        .width(IW),
        .speed(lau_pkg::FAST)
    ) u_re_sq (
        .X(real_il),
        .P(real_sq)
    );

    SqrSgn #(
        .width(IW),
        .speed(lau_pkg::FAST)
    ) u_im_sq (
        .X(imag_il),
        .P(imag_sq)
    );

    // Sum and scale
    // MAY NEED TO CHANGE LATER: it all depends on matching the model, 
    // some implementations divide by FFT_SIZE but ours doesnt
    // this shift is purely for reducing size so our accumulators later
    // aren't crazy wide but we'll prob need to test this against python model
    assign sum_full  = {1'b0, real_sq} + {1'b0, imag_sq};  // 37-bit
    assign power_ol = sum_full[2*IW:SHIFT];  // drop bottom SHIFT bits to get to 31 bit width

    always_ff @(posedge clk) begin
        valid_ol <= valid_il;
    end

endmodule
