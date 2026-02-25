module mac_unit #(
    parameter int POWER_W = 31,
    parameter int COEFF_W = 16,
    parameter int ACCUM_W = 54
)(
    input  logic                 clk_i,
    input  logic                 reset_i,
    input  logic [POWER_W-1:0]   power_i,
    input  logic [COEFF_W-1:0]   weight_i,
    input  logic                 accumulate_i,
    input  logic                 clear_i,
    output logic [ACCUM_W-1:0]   accum_o
);
    logic [POWER_W+COEFF_W-1:0] product;

    // MulUns IP for the multiply
    MulUns #(
        .widthX(POWER_W),
        .widthY(COEFF_W),
        .speed(2)
    ) u_mul (
        .X(power_i),
        .Y(weight_i),
        .P(product)
    );

    // accumulator
    always_ff @(posedge clk_i) begin
        if (reset_i || clear_i)
            accum_o <= '0;
        else if (accumulate_i)
            accum_o <= accum_o + {{(ACCUM_W-(POWER_W+COEFF_W)){1'b0}}, product};
    end

endmodule