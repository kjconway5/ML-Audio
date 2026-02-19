module log_lut #(
    parameter IN_W = 54, // accumulator width from filterbank
    parameter OUT_W = 16, // output width to CNN
    parameter FRAC_BITS = 6 // LUT has 2^FRAC_BITS entries (64 here)
)();

endmodule