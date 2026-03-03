module mel_filterbank #(
    parameter int N_MELS     = 40,
    parameter int N_BINS     = 129,
    parameter int MAX_COEFFS = 16,
    parameter int POWER_W    = 31,
    parameter int WEIGHT_W   = 16,
    parameter int ACCUM_W    = 54

)(
    input  [0:0]               clk_i,
    input  [0:0]               reset_i,
    input  logic [POWER_W-1:0]       power_il,
    input  logic [0:0]               valid_il,
    output logic [ACCUM_W-1:0]       mel_ol [N_MELS],
    output logic [0:0]               valid_ol
);

logic valid_ol_r;
//Mac Control
logic [7:0] bin_counter;        
logic [3:0] coeff_idx [N_MELS]; 
logic [0:0]    active    [N_MELS]; 
logic [0:0] clear_i;
//Mel_Coeff
logic [7:0] start_bin  [N_MELS];
logic [7:0] end_bin    [N_MELS];
logic [WEIGHT_W-1:0] weight_out [N_MELS];

//Raw accumulator outputs from MACs (before output capture)
logic [ACCUM_W-1:0] accum_raw [N_MELS];

//40 ROM "Instances" 1 Per bin
//Hopefully synthesizes to one ROM
generate
    for (genvar m = 0; m < N_MELS; m++) begin : gen_rom
        mel_coeff_rom #(
            .MEL_BINS   (N_MELS),
            .MAX_COEFFS (MAX_COEFFS),
            .COEFF_W    (WEIGHT_W),
            .BIN_W      (8)
        ) u_rom (
            .mel_idx   (m[5:0]),           // constant per instance
            .coeff_idx (coeff_idx[m]),     // driven by controller
            .weight_out(weight_out[m]),    // feeds MAC
            .start_bin (start_bin[m]),     // feeds active/coeff_idx logic
            .end_bin   (end_bin[m])        // feeds active logic
        );
    end
endgenerate
//Generate 40 Macs

generate
    for(genvar n = 0; n < N_MELS; n++) begin : gen_mac
    mac_unit # (
        .POWER_W(POWER_W),
        .COEFF_W(WEIGHT_W),
        .ACCUM_W(ACCUM_W)

    ) gen_macs (
        .clk_i(clk_i),
        .reset_i(reset_i),
        .power_i(power_il),
        .weight_i(weight_out[n]),
        .accumulate_i(active[n] && valid_il),
        .clear_i(clear_i),
        .accum_o(accum_raw[n])
    );
    end
endgenerate

//Output capture: latch accumulator values on the clear_i edge,
//before the MACs zero themselves (both use pre-edge values via NBA).
generate
    for (genvar n = 0; n < N_MELS; n++) begin : gen_output_reg
        always_ff @(posedge clk_i) begin
            if (reset_i)
                mel_ol[n] <= '0;
            else if (clear_i)
                mel_ol[n] <= accum_raw[n];
        end
    end
endgenerate

//MAC controller
generate
    for( genvar m = 0; m < N_MELS; m++) begin
        assign coeff_idx[m] = bin_counter - start_bin[m];
        assign active[m] = (bin_counter >= start_bin[m]) && (bin_counter <= end_bin[m]);
    end
endgenerate

//Bin Counter & MACS reset
always_ff @(posedge clk_i ) begin : bin_ctrl
    clear_i <= 1'b0;
    
    if(reset_i) begin
        bin_counter <= '0;
        clear_i <= 1'b1;
    end
    else if(valid_il) begin
        if(bin_counter == N_BINS - 1) begin
        bin_counter <= '0;  //Circular Counter
        clear_i <= 1'b1;    //Clear MACS after new frame
        end
        else begin
            bin_counter <= bin_counter + 1'b1;
        end

    end

end
//Pulse valid_ol one cycle after clear_i (so mel_ol has time to latch),
//but suppress the spurious pulse that clear_i generates during reset.
always_ff @(posedge clk_i) begin
    if (reset_i)
        valid_ol_r <= 1'b0;
    else
        valid_ol_r <= clear_i;
end

assign valid_ol = valid_ol_r;


endmodule