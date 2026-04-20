module requant #(
    parameter ACC_W  = 32,
    parameter DATA_W = 8
)(
    input  wire signed [ACC_W-1:0]   acc,
    input  wire        [31:0]        mult,     // Q0.32 fixed-point multiplier; effective scale = mult * 2^-(shift+32)
    input  wire        [4:0]         shift,    // exponent n: right-shift applied after extracting upper 32 bits of product
    input  wire                      relu_en,  // 1 for all layers except classifier
    output wire signed [DATA_W-1:0]  out
);

    // ── Legacy power-of-2 shift (commented out — uncomment to revert) ──────────
    // To fall back: comment out the multiply-shift block below, uncomment these
    // two lines, and change `saturated` to reference `shifted` instead of `scaled`.
    //
    // wire signed [ACC_W-1:0] shifted = acc >>> shift;
    //
    // wire signed [DATA_W-1:0] saturated =
    //     (shifted > 32'sh0000007F) ?  8'sh7F :
    //     (shifted < -32'sh00000080) ? -8'sh80 :
    //     shifted[DATA_W-1:0];

    // ── Multiply-shift requantization (ARM GEMMLOWP / TFLite Micro style) ──────
    // Effective scale applied to acc:  S = mult * 2^-(shift+32)
    //
    // Step 1: 32-bit signed × 32-bit unsigned → 64-bit signed product.
    //         {1'b0, mult} zero-extends mult to 33 bits so the operator sees
    //         a signed×signed multiply with mult always treated as positive.
    wire signed [63:0] product = $signed(acc) * $signed({1'b0, mult});
    //
    // Step 2: Extract upper 32 bits of the 64-bit product (implicit ÷2^32),
    //         then arithmetic right-shift by n to complete the scale.
    wire signed [31:0] scaled  = product[63:32] >>> shift;

    // ── Saturate 32 → 8 bits ──────────────────────────────────────────────────
    wire signed [DATA_W-1:0] saturated =
        (scaled > 32'sh0000007F) ?  8'sh7F :
        (scaled < -32'sh00000080) ? -8'sh80 :
        scaled[DATA_W-1:0];

    // If relu_en and result is negative, clamp to zero
    assign out = (relu_en && saturated[DATA_W-1]) ? 8'sh00 : saturated;

endmodule
