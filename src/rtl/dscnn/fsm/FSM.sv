// FSM.sv

module FSM #(
    parameter DATA_W  = 8,
    parameter ACC_W   = 32,
    parameter ADDR_W  = 14,
    parameter SPECT_AW = 11   // matches spectrogram_sram ADDR_W
)(
    input  wire                     clk,
    input  wire                     reset,

    // SERV config signals
    input  wire                     cfg_we,
    input  wire [7:0]               cfg_addr,   // see address map above
    input  wire [7:0]               cfg_wdata,  // Handled by UART data input
                                                // During Verification use python input mimicing flash input
    // Spectrogram handshake signals
    input  wire                     spect_done,       // new spectrogram ready for reading
    input  wire                     spect_write_sel,  // which bank preprocessor just finished writing

    // Spectrogram SRAM read port
    output wire [SPECT_AW-1:0]      sp_raddr,   // 
    input  wire signed [DATA_W-1:0] sp_a_rdata,   // bank A read data
    input  wire signed [DATA_W-1:0] sp_b_rdata,   // bank B read data

    // DSCNN signals
    input  wire                     start,   // ignored unless cfg_load_done & spect_ready
    output reg                      done,
    output reg  [2:0]               class_out,

    // Weight SRAM signals
    output wire [12:0]              w_addr,   
    input  wire signed [DATA_W-1:0] w_data,

    // Weight writing signals
    input wire                      weights_ready, // from UART control
    output wire                     inference_idle, // to UART

    // Bias ROM interface
    output wire [7:0]               bias_addr,
    input  wire signed [31:0]       bias_data,

    //Feature SRAM signals
    // Bank A
    output reg                      fs_a_we,
    output reg  [ADDR_W-1:0]        fs_a_waddr,
    output reg  signed [DATA_W-1:0] fs_a_wdata,
    output wire [ADDR_W-1:0]        fs_a_raddr,   
    input  wire signed [DATA_W-1:0] fs_a_rdata,
    // Bank B
    output reg                      fs_b_we,
    output reg  [ADDR_W-1:0]        fs_b_waddr,
    output reg  signed [DATA_W-1:0] fs_b_wdata,
    output wire [ADDR_W-1:0]        fs_b_raddr,   
    input  wire signed [DATA_W-1:0] fs_b_rdata,

    // MAC signals (scalar: one INT8 × INT8 per cycle)
    output reg                      mac_en,
    output reg                      mac_clear,
    output reg  signed [DATA_W-1:0] mac_ifmap,
    output reg  signed [DATA_W-1:0] mac_weight,
    output reg  signed [ACC_W-1:0]  mac_bias,
    input  wire signed [ACC_W-1:0]  mac_acc,

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

    reg        cfg_load_done;   // set by UART control

    // Config decoder signals
    reg [3:0] cfg_layer;
    reg [3:0] cfg_field;

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
                cfg_layer = cfg_addr[7:4];
                cfg_field = cfg_addr[3:0];
                case (cfg_field)
                    4'd0:  cfg_in_ch   [cfg_layer] <= cfg_wdata;
                    4'd1:  cfg_out_ch  [cfg_layer] <= cfg_wdata;
                    4'd2:  cfg_kH      [cfg_layer] <= cfg_wdata[3:0];
                    4'd3:  cfg_kW      [cfg_layer] <= cfg_wdata[3:0];
                    4'd4:  cfg_stride_h[cfg_layer] <= cfg_wdata[1:0];
                    4'd5:  cfg_stride_w[cfg_layer] <= cfg_wdata[1:0];
                    4'd6:  cfg_pad_h   [cfg_layer] <= cfg_wdata[3:0];
                    4'd7:  cfg_pad_w   [cfg_layer] <= cfg_wdata[3:0];
                    4'd8:  cfg_dw      [cfg_layer] <= cfg_wdata[0];
                    4'd9:  cfg_w_off   [cfg_layer][7:0]  <= cfg_wdata;       // Weight offset split into two (bot 8 bits)
                    4'd10: cfg_w_off   [cfg_layer][12:8] <= cfg_wdata[4:0];  // high 5 bits
                    4'd11: cfg_shift   [cfg_layer] <= cfg_wdata[4:0];
                    4'd12: cfg_relu    [cfg_layer] <= cfg_wdata[0];
                    4'd13: cfg_ofmap_h [cfg_layer] <= cfg_wdata;
                    4'd14: cfg_ofmap_w [cfg_layer] <= cfg_wdata;
                    4'd15: cfg_bias_off[cfg_layer] <= cfg_wdata;
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

    // FSM states
    localparam  IDLE        = 4'd0,
                LOAD_LAYER  = 4'd1,
                CLEAR_ACC   = 4'd2,
                FETCH       = 4'd3,   // 1 cycle latency state to match read latency of SRAM
                COMPUTE     = 4'd4,   // read SRAM data, accumulate MAC
                DRAIN       = 4'd5,   // flush last MAC product before WRITE_OFMAP
                WRITE_OFMAP = 4'd6,
                NEXT_PIXEL  = 4'd7,
                NEXT_LAYER  = 4'd8,
                GLOBAL_POOL = 4'd9,   // global-average-pool argmax scan
                OUTPUT      = 4'd10;  // output class_out and assert done

    reg [3:0]  state;
    reg [3:0]  layer;       // FSM layer counter (0-9), separate from cfg_layer decode above
    reg        buf_sel;     // feature SRAM ping-pong: 0=read A write B, 1=read B write A
    reg [7:0]  oh, ow, oc;
    reg [7:0]  ic;
    reg [3:0]  kh, kw;

    reg signed [ACC_W-1:0] max_val;
    reg [2:0]              max_idx;

    // Global average pool accumulators 
    reg signed [ACC_W-1:0] global_pool_acc [0:6];
    reg [2:0]              global_pool_idx;

    reg signed [DATA_W-1:0] ifmap_val;  


    reg signed [8:0]  comb_ih_raw;
    reg signed [8:0]  comb_iw_raw;
    reg               comb_in_bounds;
    reg [7:0]         comb_ifmap_h;
    reg [7:0]         comb_ifmap_w;
    reg [ADDR_W-1:0]  comb_feat_addr;
    reg [SPECT_AW-1:0] comb_sp_raddr;
    reg [12:0]        comb_w_addr;

    always @(*) begin
        // Determine input feature-map dimensions for current layer
        if (layer == 4'd0) begin
            comb_ifmap_h = 8'd50;
            comb_ifmap_w = 8'd40;
        end else begin
            comb_ifmap_h = cfg_ofmap_h[layer - 1];
            comb_ifmap_w = cfg_ofmap_w[layer - 1];
        end

        // Input spatial coordinates
        comb_ih_raw = ($signed({1'b0, oh}) * $signed({2'b0, cfg_stride_h[layer]}))
                    + $signed({1'b0, kh})
                    - $signed({1'b0, cfg_pad_h[layer]});
        comb_iw_raw = ($signed({1'b0, ow}) * $signed({2'b0, cfg_stride_w[layer]}))
                    + $signed({1'b0, kw})
                    - $signed({1'b0, cfg_pad_w[layer]});

        comb_in_bounds = (comb_ih_raw >= 9'sh0)
                      && (comb_ih_raw < $signed({1'b0, comb_ifmap_h}))
                      && (comb_iw_raw >= 9'sh0)
                      && (comb_iw_raw < $signed({1'b0, comb_ifmap_w}));

        // Spectrogram read address (layer 0 only)
        comb_sp_raddr = (comb_in_bounds)
            ? ($unsigned(comb_ih_raw[7:0]) * 8'd40 + $unsigned(comb_iw_raw[7:0]))
            : {SPECT_AW{1'b0}};

        // Feature SRAM read address (layers 1-9)
        comb_feat_addr = (comb_in_bounds)
            ? ((cfg_dw[layer] ? oc : ic) * comb_ifmap_h * comb_ifmap_w
               + $unsigned(comb_ih_raw[7:0]) * comb_ifmap_w
               + $unsigned(comb_iw_raw[7:0]))
            : {ADDR_W{1'b0}};

        // Weight SRAM read address
        if (cfg_dw[layer])
            comb_w_addr = cfg_w_off[layer]
                        + oc * cfg_kH[layer] * cfg_kW[layer]
                        + kh * cfg_kW[layer] + kw;
        else
            comb_w_addr = cfg_w_off[layer]
                        + oc * cfg_in_ch[layer] * cfg_kH[layer] * cfg_kW[layer]
                        + ic * cfg_kH[layer] * cfg_kW[layer]
                        + kh * cfg_kW[layer] + kw;
    end

    // Drive SRAM read address ports combinationally
    assign sp_raddr   = comb_sp_raddr;
    assign w_addr     = comb_w_addr;
    assign fs_a_raddr = comb_feat_addr;
    assign fs_b_raddr = comb_feat_addr;

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
            global_pool_idx <= 3'd0;
            for (int i = 0; i < 7; i = i + 1)
                global_pool_acc[i] <= {ACC_W{1'b0}};
        end else begin
            mac_en    <= 1'b0;
            mac_clear <= 1'b0;
            fs_a_we   <= 1'b0;
            fs_b_we   <= 1'b0;
            done      <= 1'b0;

            case (state)

                IDLE: begin
                    // Is spectrogram and config loading ready
                    if (start && cfg_load_done && spect_ready && weights_ready) begin
                        layer   <= 4'd0;
                        buf_sel <= 1'b0;
                        state   <= LOAD_LAYER;
                    end
                end

                LOAD_LAYER: begin
                    rq_shift   <= cfg_shift[layer];
                    rq_relu_en <= cfg_relu[layer];
                    oh <= 8'd0; ow <= 8'd0; oc <= 8'd0;
                    // Zero the GAP accumulators at the start of the classifier layer
                    if (layer == N_LAYERS-1) begin
                        for (int i = 0; i < 7; i = i + 1)
                            global_pool_acc[i] <= {ACC_W{1'b0}};
                    end
                    state <= CLEAR_ACC;
                end

                CLEAR_ACC: begin
                    mac_bias  <= bias_data;
                    mac_clear <= 1'b1;
                    ic <= 8'd0; kh <= 4'd0; kw <= 4'd0;
                    state <= FETCH;
                end

                FETCH: begin
                    state <= COMPUTE;
                end

                
                COMPUTE: begin
                    // Select ifmap value (zero for out-of-bounds / padding)
                    if (!comb_in_bounds) begin
                        ifmap_val = {DATA_W{1'b0}};
                    end else if (layer == 4'd0) begin
                        ifmap_val = sp_rdata;
                    end else begin
                        ifmap_val = (!buf_sel) ? fs_a_rdata : fs_b_rdata;
                    end

                    // Accumulate — always enabled; DRAIN flushes the last product
                    mac_en     <= 1'b1;
                    mac_ifmap  <= ifmap_val;
                    mac_weight <= w_data;

                    // Advance kernel position counters
                    if (kw < cfg_kW[layer]-1) begin
                        kw    <= kw + 1;
                        state <= FETCH;          // pre-fetch next position
                    end else begin
                        kw <= 4'd0;
                        if (kh < cfg_kH[layer]-1) begin
                            kh    <= kh + 1;
                            state <= FETCH;
                        end else begin
                            kh <= 4'd0;
                            if (!cfg_dw[layer] && ic < cfg_in_ch[layer]-1) begin
                                ic    <= ic + 1;
                                state <= FETCH;
                            end else begin
                                // Last kernel element — drain before writing output
                                state <= DRAIN;
                            end
                        end
                    end
                end

     
                DRAIN: begin
                    state <= WRITE_OFMAP;
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

                    // Global average pool: accumulate INT8 outputs across the 25×20 spatial map.
                    if (layer == N_LAYERS-1) begin
                        global_pool_acc[oc[2:0]] <= global_pool_acc[oc[2:0]]
                                                  + {{24{rq_out[7]}}, rq_out};
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
                    if (layer == N_LAYERS-1) begin
                        // Classifier done — enter global pool argmax scan
                        global_pool_idx <= 3'd0;
                        state           <= GLOBAL_POOL;
                    end else begin
                        layer <= layer + 1;
                        state <= LOAD_LAYER;
                    end
                end

                // Scan the 7 global_pool_acc entries and find the winning class.
                // Takes exactly 7 cycles (one per class), then proceeds to OUTPUT.
                GLOBAL_POOL: begin
                    if (global_pool_idx == 3'd0 ||
                            $signed(global_pool_acc[global_pool_idx]) > $signed(max_val)) begin
                        max_val <= global_pool_acc[global_pool_idx];
                        max_idx <= global_pool_idx;
                    end
                    if (global_pool_idx == 3'd6)
                        state <= OUTPUT;
                    else
                        global_pool_idx <= global_pool_idx + 1;
                end

                OUTPUT: begin
                    class_out <= max_idx;
                    done      <= 1'b1;
                    state     <= IDLE;
                end

            endcase
        end
    end

    assign inference_idle = (state == IDLE);

    // Bias address: base offset for current layer + current output channel
    assign bias_addr = cfg_bias_off[layer] + oc;

endmodule
