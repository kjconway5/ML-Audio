// AXI-Stream RX interface: byte_valid_i stays high until byte_ready_o.
// Host sends each sample as 2 bytes, little-endian:
//   byte 0: sample[7:0]   (low byte)
//   byte 1: sample[15:8]  (high byte, only bits [13:8] used)

module sample_unpacker (
    input  logic        clk_i,
    input  logic        reset_i,
    input  logic [7:0]  byte_i,
    input  logic        byte_valid_i,
    output logic        byte_ready_o,
    output logic [13:0] sample_o,
    output logic        sample_valid_o
);

    logic [7:0] low_byte;
    logic       have_low;

    // Always accept bytes immediately
    assign byte_ready_o = 1'b1;

    always_ff @(posedge clk_i) begin
        if (reset_i) begin
            have_low       <= 0;
            sample_valid_o <= 0;
        end else begin
            sample_valid_o <= 0;

            if (byte_valid_i && byte_ready_o) begin
                if (!have_low) begin
                    low_byte <= byte_i;
                    have_low <= 1;
                end else begin
                    sample_o       <= {byte_i[5:0], low_byte};
                    sample_valid_o <= 1;
                    have_low       <= 0;
                end
            end
        end
    end

endmodule
