// FSM.sv

// MMIO cfg_addr map (Lower 4-bits address field (16 entries - 4-bits), upper 4-bits adresses layer  
//   Per-layer fields: base = layer * 16 (computation for base adress based on number of fields per layer) 
//   +0  cfg_in_ch       +1  cfg_out_ch      +2  cfg_kH          +3  cfg_kW
//   +4  cfg_stride_h    +5  cfg_stride_w    +6  cfg_pad_h       +7  cfg_pad_w
//   +8  cfg_dw[0]       +9  cfg_w_off[7:0]  +10 cfg_w_off[12:8] +11 cfg_shift
//   +12 cfg_relu        +13 cfg_ofmap_h     +14 cfg_ofmap_w     +15 cfg_bias_off

//   Special address: 8'hFF → write 1 to assert cfg_load_done internally

module FSM #(
    parameter N_MACS  = 16,
    parameter DATA_W  = 8,
    parameter ACC_W   = 32,
    parameter ADDR_W  = 14,
    parameter SPECT_AW = 11   // matches spectrogram_sram ADDR_W (covers 0-2047)
)(
    input  wire                     clk,
    input  wire                     reset,

    // SERV config signals 
    input  wire                     cfg_we,
    input  wire [7:0]               cfg_addr,   // see address map above
    input  wire [7:0]               cfg_wdata,  // Handled by SPI/Flash data input 
                                                // During Verification use python input mimicing flash input 
    // Spectrogram handshake signals 
    input  wire                     spect_done,       // new spectrogram ready for reading
    input  wire                     spect_write_sel,  // which bank preprocessor just finished writing

    // Spectrogram SRAM read port 
    output reg  [SPECT_AW-1:0]      sp_raddr,
    input  wire signed [DATA_W-1:0] sp_a_rdata,   // bank A read data
    input  wire signed [DATA_W-1:0] sp_b_rdata,   // bank B read data

    // DSCNN signals
    input  wire                     start,   // ignored unless cfg_load_done & spect_ready
    output reg                      done,
    output reg  [2:0]               class_out,

    // Weight SRAM signals 
    output reg  [12:0]              w_addr,
    input  wire signed [DATA_W-1:0] w_data,

    //Feature SRAM signals 
    // Bank A 
    output reg                      fs_a_we,
    output reg  [ADDR_W-1:0]        fs_a_waddr,
    output reg  signed [DATA_W-1:0] fs_a_wdata,
    output reg  [ADDR_W-1:0]        fs_a_raddr,
    input  wire signed [DATA_W-1:0] fs_a_rdata,
    // Bank B 
    output reg                      fs_b_we,
    output reg  [ADDR_W-1:0]        fs_b_waddr,
    output reg  signed [DATA_W-1:0] fs_b_wdata,
    output reg  [ADDR_W-1:0]        fs_b_raddr,
    input  wire signed [DATA_W-1:0] fs_b_rdata,

    // Mac Array signals 
    output reg                      mac_en,
    output reg                      mac_clear,
    output reg  signed [DATA_W-1:0] mac_ifmap  [0:N_MACS-1],
    output reg  signed [DATA_W-1:0] mac_weight [0:N_MACS-1],
    output reg  signed [ACC_W-1:0]  mac_bias,
    input  wire signed [ACC_W-1:0]  mac_acc,
    input  wire                     mac_valid,

    // Requant module signals 
    output reg  [4:0]               rq_shift,
    output reg                      rq_relu_en,
    input  wire signed [DATA_W-1:0] rq_out
);

    // Model Config Parameters 
    localparam N_LAYERS = 10;

    reg [7:0]  cfg_in_ch    [0:N_LAYERS-1];
    reg [7:0]  cfg_out_ch   [0:N_LAYERS-1];
    reg [3:0]  cfg_kH       [0:N_LAYERS-1];
    reg [3:0]  cfg_kW       [0:N_LAYERS-1];
    reg [1:0]  cfg_stride_h [0:N_LAYERS-1];
    reg [1:0]  cfg_stride_w [0:N_LAYERS-1];
    reg [3:0]  cfg_pad_h    [0:N_LAYERS-1];
    reg [3:0]  cfg_pad_w    [0:N_LAYERS-1];
    reg        cfg_dw       [0:N_LAYERS-1];
    reg [12:0] cfg_w_off    [0:N_LAYERS-1];
    reg [4:0]  cfg_shift    [0:N_LAYERS-1];
    reg        cfg_relu     [0:N_LAYERS-1];
    reg [7:0]  cfg_ofmap_h  [0:N_LAYERS-1];
    reg [7:0]  cfg_ofmap_w  [0:N_LAYERS-1];
    reg [7:0]  cfg_bias_off [0:N_LAYERS-1];

    reg        cfg_load_done;   // set by SERV writing 1 to addr 8'hFF

    // Config decoder signals 
    reg [3:0] layer;
    reg [3:0] field;

    // MMIO write decode
    always @(posedge clk) begin
        if (reset) begin
            cfg_load_done <= 1'b0;
        end else if (cfg_we) begin
            if (cfg_addr == 8'hFF) begin
                cfg_load_done <= 1'b1;
            end else begin
                // Decode layer index and field offset from address
                // layer  = cfg_addr[7:4]  upper 4-bits (layers 0-9) 
                // field  = cfg_addr[3:0]  lower 4-bits (fields 0-15) 
                layer = cfg_addr[7:4];
                field = cfg_addr[3:0];
                case (field)
                    4'd0:  cfg_in_ch   [layer] <= cfg_wdata;
                    4'd1:  cfg_out_ch  [layer] <= cfg_wdata;
                    4'd2:  cfg_kH      [layer] <= cfg_wdata[3:0];
                    4'd3:  cfg_kW      [layer] <= cfg_wdata[3:0];
                    4'd4:  cfg_stride_h[layer] <= cfg_wdata[1:0];
                    4'd5:  cfg_stride_w[layer] <= cfg_wdata[1:0];
                    4'd6:  cfg_pad_h   [layer] <= cfg_wdata[3:0];
                    4'd7:  cfg_pad_w   [layer] <= cfg_wdata[3:0];
                    4'd8:  cfg_dw      [layer] <= cfg_wdata[0];
                    4'd9:  cfg_w_off   [layer][7:0]  <= cfg_wdata;       // Weight offset split into two (bot 8 bits) 
                    4'd10: cfg_w_off   [layer][12:8] <= cfg_wdata[4:0];  // high 5 bits
                    4'd11: cfg_shift   [layer] <= cfg_wdata[4:0];
                    4'd12: cfg_relu    [layer] <= cfg_wdata[0];
                    4'd13: cfg_ofmap_h [layer] <= cfg_wdata;
                    4'd14: cfg_ofmap_w [layer] <= cfg_wdata;
                    4'd15: cfg_bias_off[layer] <= cfg_wdata;
                    default: ;
                endcase
            end
        end
    end

    
    reg spect_ready;      // at least one complete spectrogram available
    reg spect_read_sel;   // which bank FSM reads from (0=A, 1=B)

    // Spectrogram Ping-Pong buffer logic 
    always @(posedge clk) begin
        if (reset) begin
            spect_ready    <= 1'b0;
            spect_read_sel <= 1'b0;
        end else if (spect_done) begin
            spect_read_sel <= spect_write_sel;  // read from bank that just finished writing 
            spect_ready    <= 1'b1;
        end
    end

    // select spectrogram read data from the correct bank 
    wire signed [DATA_W-1:0] sp_rdata =
        (spect_read_sel == 1'b0) ? sp_a_rdata : sp_b_rdata;

    // FSM 
    localparam  IDLE        = 3'd0,
                LOAD_LAYER  = 3'd1,
                CLEAR_ACC   = 3'd2,
                COMPUTE     = 3'd3,
                WRITE_OFMAP = 3'd4,
                NEXT_PIXEL  = 3'd5,
                NEXT_LAYER  = 3'd6,
                OUTPUT      = 3'd7;

    reg [2:0]  state;
    reg [3:0]  layer;
    reg        buf_sel;     // feature SRAM ping-pong: 0=read A write B, 1=read B write A
    reg [7:0]  oh, ow, oc;
    reg [7:0]  ic;
    reg [3:0]  kh, kw;
    reg [3:0]  mac_idx;

    reg signed [ACC_W-1:0] max_val;
    reg [2:0]              max_idx;

    //
    reg signed [8:0]        ih_raw;     //ih_raw = oh * stride_h + kh - pad_h (If negative = In padding bits) 
    reg signed [8:0]        iw_raw;     //iw_raw = ow * stride_w + kw - pad_w (If negative = In padding bits) 
    reg [7:0]               ifmap_h;    // input feature map height for current layer
    reg [7:0]               ifmap_w;    // input feature map width for current layer
    reg                     in_bounds;  // is pixel within bounds of feature map (if not will feed 0 to MAC) 
    reg [ADDR_W-1:0]        feat_addr;  // feat_addr = ic * H * W + ih * W + iw (flat SRAM feature addr) 
    reg signed [DATA_W-1:0] ifmap_val;  // ifmap sample routed to MAC (0 if padding)

    always @(posedge clk) begin
        if (reset) begin
            state     <= IDLE;
            done      <= 1'b0;
            layer     <= 4'd0;
            buf_sel   <= 1'b0;
            mac_en    <= 1'b0;
            mac_clear <= 1'b0;
            fs_a_we   <= 1'b0;
            fs_b_we   <= 1'b0;
            sp_raddr  <= {SPECT_AW{1'b0}};
        end else begin
            mac_en    <= 1'b0;
            mac_clear <= 1'b0;
            fs_a_we   <= 1'b0;
            fs_b_we   <= 1'b0;
            done      <= 1'b0;

            case (state)

                IDLE: begin
                    // Is spectrogram and config loading ready 
                    if (start && cfg_load_done && spect_ready) begin
                        layer   <= 4'd0;
                        buf_sel <= 1'b0;   
                        state   <= LOAD_LAYER;
                    end
                end

                LOAD_LAYER: begin
                    rq_shift   <= cfg_shift[layer];
                    rq_relu_en <= cfg_relu[layer];
                    oh <= 8'd0; ow <= 8'd0; oc <= 8'd0;
                    state <= CLEAR_ACC;
                end

                CLEAR_ACC: begin
                    mac_bias  <= 32'sh0;
                    mac_clear <= 1'b1;
                    ic <= 8'd0; kh <= 4'd0; kw <= 4'd0; mac_idx <= 4'd0;
                    state <= COMPUTE;
                end

                COMPUTE: begin
                    // Compute input (ifmap) spatial coordinates
                    ih_raw = ($signed({1'b0, oh}) * $signed({2'b0, cfg_stride_h[layer]})) //ih_raw = oh * stride_h + kh - pad_h
                           + $signed({1'b0, kh})
                           - $signed({1'b0, cfg_pad_h[layer]});
                    iw_raw = ($signed({1'b0, ow}) * $signed({2'b0, cfg_stride_w[layer]})) //iw_raw = ow * stride_w + kw - pad_w
                           + $signed({1'b0, kw})
                           - $signed({1'b0, cfg_pad_w[layer]});

                    // Layer 0 input: 50 rows × 40 cols (# of frames x # of N_MELS) 
                    if (layer == 4'd0) begin
                        ifmap_h = 8'd50;
                        ifmap_w = 8'd40;
                    end else begin
                        ifmap_h = cfg_ofmap_h[layer - 1]; // ifmap_h/w based on ofmap_h/w of previous layer 
                        ifmap_w = cfg_ofmap_w[layer - 1];
                    end

                    in_bounds = (ih_raw >= 9'sh0)
                             && (ih_raw < $signed({1'b0, ifmap_h})) // ifmap_h is unsigned reg (need $signed to compute -ih_raw < ifmap_h) 
                             && (iw_raw >= 9'sh0)
                             && (iw_raw < $signed({1'b0, ifmap_w}));

                    // Compute spectrogram addr (Row * 40 + Column) Row-Col indexing 
                    if (layer == 4'd0) begin                        
                        sp_raddr <= in_bounds       // if in_bounds -> can treat as unsigned
                            ? ($unsigned(ih_raw[7:0]) * 8'd40 + $unsigned(iw_raw[7:0]))
                            : {SPECT_AW{1'b0}};     
                    end else begin
                        // Compute feature SRAM addr = ic*H*W + ih*W + iw
                        feat_addr = in_bounds
                            ? (ic * ifmap_h * ifmap_w
                               + $unsigned(ih_raw[7:0]) * ifmap_w
                               + $unsigned(iw_raw[7:0]))
                            : {ADDR_W{1'b0}};
                        if (!buf_sel) fs_a_raddr <= feat_addr;
                        else          fs_b_raddr <= feat_addr;
                    end

                    // Feed MAC_array 
                    mac_en <= 1'b1;

                    // Decide where to read from to feed MAC 
                    if (!in_bounds) begin
                        ifmap_val = {DATA_W{1'b0}};
                    end else if (layer == 4'd0) begin
                        ifmap_val = sp_rdata;
                    end else begin
                        ifmap_val = (!buf_sel) ? fs_a_rdata : fs_b_rdata;
                    end

                    // Compute Weight Address
                    // Depthwise: base + (ic*kH*kW + kh*kW + kw)
                    // (pointwise: base + (oc*in_ch*kH*kW + ic*kH*kW + kh*kW + kw)
                    if (cfg_dw[layer])    // depthwise 
                        w_addr <= cfg_w_off[layer]                      // Row-Major flattened memory indexing
                                + ic * cfg_kH[layer] * cfg_kW[layer]
                                + kh * cfg_kW[layer] + kw;
                    else
                        w_addr <= cfg_w_off[layer]  // pointwise
                                + oc * cfg_in_ch[layer] * cfg_kH[layer] * cfg_kW[layer]
                                + ic * cfg_kH[layer] * cfg_kW[layer]
                                + kh * cfg_kW[layer] + kw;

                    // Select which MAC lane 
                    mac_ifmap [mac_idx] <= ifmap_val;
                    mac_weight[mac_idx] <= w_data;
                    mac_idx            <= mac_idx + 1; // Counter that wraps naturally at 4'b1111 -> 4'b0000

                    // Check if entire kernel is computed 
                    if (kw == cfg_kW[layer]-1 && kh == cfg_kH[layer]-1 &&
                        ic == cfg_in_ch[layer]-1) begin
                        mac_en <= 1'b0;
                        state  <= WRITE_OFMAP;
                    end else begin  // increment kernel width if not at end of row 
                        if (kw < cfg_kW[layer]-1) begin
                            kw <= kw + 1;
                        end else begin
                            kw <= 4'd0;
                            if (kh < cfg_kH[layer]-1) begin // increment kernel height if not at end of column 
                                kh <= kh + 1;
                            end else begin
                                kh <= 4'd0;
                                if (!cfg_dw[layer]) // increment ic for pointwise 
                                    ic <= ic + 1;
                            end
                        end
                    end
                end

                WRITE_OFMAP: begin
                    if (!buf_sel) begin
                        fs_b_we    <= 1'b1;
                        fs_b_waddr <= oc * cfg_ofmap_h[layer] * cfg_ofmap_w[layer]
                                    + oh * cfg_ofmap_w[layer] + ow;
                        fs_b_wdata <= rq_out;
                    end else begin
                        fs_a_we    <= 1'b1;
                        fs_a_waddr <= oc * cfg_ofmap_h[layer] * cfg_ofmap_w[layer]
                                    + oh * cfg_ofmap_w[layer] + ow;
                        fs_a_wdata <= rq_out;
                    end

                    // Argmax for classifier layer
                    if (layer == 4'd9) begin
                        if (oc == 8'd0 || $signed(mac_acc) > $signed(max_val)) begin
                            max_val <= mac_acc;
                            max_idx <= oc[2:0];
                        end
                    end

                    state <= NEXT_PIXEL;
                end

                NEXT_PIXEL: begin
                    if (ow < cfg_ofmap_w[layer]-1) begin
                        ow    <= ow + 1;
                        state <= CLEAR_ACC;
                    end else begin
                        ow <= 8'd0;
                        if (oh < cfg_ofmap_h[layer]-1) begin
                            oh    <= oh + 1;
                            state <= CLEAR_ACC;
                        end else begin
                            oh <= 8'd0;
                            if (oc < cfg_out_ch[layer]-1) begin
                                oc    <= oc + 1;
                                state <= CLEAR_ACC;
                            end else begin
                                oc    <= 8'd0;
                                state <= NEXT_LAYER;
                            end
                        end
                    end
                end

                NEXT_LAYER: begin
                    buf_sel <= ~buf_sel;
                    if (layer == N_LAYERS-1)
                        state <= OUTPUT;
                    else begin
                        layer <= layer + 1;
                        state <= LOAD_LAYER;
                    end
                end

                OUTPUT: begin
                    class_out <= max_idx;
                    done      <= 1'b1;
                    state     <= IDLE;
                end

            endcase
        end
    end

endmodule