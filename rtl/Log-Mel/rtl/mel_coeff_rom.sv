/*
NON-SPARSE IMPLEMENTATION: CHANGE LATER FOR MEMORY SAVING
*/
module mel_coeff_rom #(
    parameter MEL_BINS  = 40,
    parameter MAX_COEFFS = 12,   // max non-zero weights per filter
    parameter COEFF_W   = 16,
    parameter BIN_W     = 7      // log2(128) = 7 bits to address bins
)(
    input  logic [$clog2(MEL_BINS)-1:0]   mel_idx,
    input  logic [$clog2(MAX_COEFFS)-1:0] coeff_idx,
    output logic [COEFF_W-1:0]            weight_out,
    output logic [BIN_W-1:0]              start_bin,
    output logic [BIN_W-1:0]              end_bin
);

    // The actual coefficient data lives here
    logic [COEFF_W-1:0] coeff_mem [0:MEL_BINS-1][0:MAX_COEFFS-1];
    logic [BIN_W-1:0]   start_bins [0:MEL_BINS-1];
    logic [BIN_W-1:0]   end_bins   [0:MEL_BINS-1];

    // Load from hex files at simulation/synthesis
    initial begin
        $readmemh("mel_coeffs.hex",  coeff_mem);
        $readmemh("mel_starts.hex",  start_bins);
        $readmemh("mel_ends.hex",    end_bins);
    end

    assign weight_out = coeff_mem[mel_idx][coeff_idx];
    assign start_bin  = start_bins[mel_idx];
    assign end_bin    = end_bins[mel_idx];

endmodule