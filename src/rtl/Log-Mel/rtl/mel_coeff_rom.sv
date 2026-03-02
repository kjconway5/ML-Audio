module mel_coeff_rom #(
    parameter MEL_BINS = 40,
    parameter MAX_COEFFS = 16, // max non-zero weights per filter
    parameter COEFF_W = 16,
    parameter BIN_W = 7 // log2(128) = 7 bits to address bins
)(
    input  logic [$clog2(MEL_BINS)-1:0]   mel_idx,
    input  logic [$clog2(MAX_COEFFS)-1:0] coeff_idx,
    output logic [COEFF_W-1:0]            weight_out,
    output logic [BIN_W-1:0]              start_bin,
    output logic [BIN_W-1:0]              end_bin
);

    logic [COEFF_W-1:0] coeff_mem [0:MEL_BINS*MAX_COEFFS-1];
    logic [BIN_W-1:0]   start_bins [0:MEL_BINS-1];
    logic [BIN_W-1:0]   end_bins   [0:MEL_BINS-1];

    // load from hex files at synthesis
    // may need to change later, do we want this as combinational logic or an actual RAM
    // for simulation this should work and i think should also synthesize to combinational logic
    // something to thinka about
    initial begin
        $readmemh("../data/mel_coeffs.hex", coeff_mem);
        $readmemh("../data/mel_starts.hex", start_bins);
        $readmemh("../data/mel_ends.hex",   end_bins);
    end

    assign weight_out = coeff_mem[mel_idx * MAX_COEFFS + coeff_idx];
    assign start_bin = start_bins[mel_idx];
    assign end_bin = end_bins[mel_idx];

endmodule