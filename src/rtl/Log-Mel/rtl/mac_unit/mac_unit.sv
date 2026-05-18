module mac_unit #(
    parameter int POWER_W = 32,
    parameter int COEFF_W = 16,
    parameter int ACCUM_W = 56
)(
    input  logic                 clk_i,
    input  logic                 reset_i,
    input  logic [POWER_W-1:0]   power_i,
    input  logic [COEFF_W-1:0]   weight_i,
    input  logic                 accumulate_i,
    input  logic                 clear_i,
    output logic [ACCUM_W-1:0]   accum_o,

    output logic [POWER_W-1:0]   mac_accum_test,
    input logic test_mode_audio
);
    logic [POWER_W+COEFF_W-1:0] product;

    `ifndef SYNTHESIS
        assign product = power_i * weight_i;
    `else
        MulUns #(
            .widthX(POWER_W),
            .widthY(COEFF_W),
            .speed(2)
        ) u_mul (
            .X(power_i),
            .Y(weight_i),
            .P(product)
        );
    `endif

    always_ff @(posedge clk_i) begin
        if (reset_i || clear_i)
            accum_o <= '0;
        else if (accumulate_i)
            accum_o <= accum_o + {{(ACCUM_W-(POWER_W+COEFF_W)){1'b0}}, product};
    end

    // test mode to observe at io pins
    always_ff @(posedge clk_i) begin
        if (reset_i) begin
            mac_accum_test <= 0;
        end else if (test_mode_audio) begin
            mac_accum_test <= accum_o;
        end else begin
            mac_accum_test <= 0;
        end
    end

endmodule
