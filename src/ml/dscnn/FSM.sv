module layer_controller #(
    parameter N_MACS  = 16,
    parameter DATA_W  = 8,
    parameter ACC_W   = 32,
    parameter ADDR_W  = 14
)(
    input  wire                     clk,
    input  wire                     reset,
    input  wire                     start,         // pulse to begin inference
    output reg                      done,          // pulses when inference complete
    output reg  [2:0]               class_out,     // winning class index

    // Weight SRAM interface
    output reg  [12:0]              w_addr,
    input  wire signed [DATA_W-1:0] w_data,

    // Feature SRAM interface (ping-pong)
    output reg                      fs_a_we,
    output reg  [ADDR_W-1:0]        fs_a_waddr,
    output reg  signed [DATA_W-1:0] fs_a_wdata,
    output reg  [ADDR_W-1:0]        fs_a_raddr,
    input  wire signed [DATA_W-1:0] fs_a_rdata,

    output reg                      fs_b_we,
    output reg  [ADDR_W-1:0]        fs_b_waddr,
    output reg  signed [DATA_W-1:0] fs_b_wdata,
    output reg  [ADDR_W-1:0]        fs_b_raddr,
    input  wire signed [DATA_W-1:0] fs_b_rdata,

    // MAC array interface
    output reg                      mac_en,
    output reg                      mac_clear,
    output reg  signed [DATA_W-1:0] mac_ifmap  [0:N_MACS-1],
    output reg  signed [DATA_W-1:0] mac_weight [0:N_MACS-1],
    output reg  signed [ACC_W-1:0]  mac_bias,
    input  wire signed [ACC_W-1:0]  mac_acc,
    input  wire                     mac_valid,

    // Requant interface
    output reg  [4:0]               rq_shift,
    output reg                      rq_relu_en,
    input  wire signed [DATA_W-1:0] rq_out
);


    localparam N_LAYERS = 10;       //1 first conv, 8 - depthwise & pointwise for 4 layers, 1 classifier layer 

    reg [7:0]  cfg_in_ch    [0:N_LAYERS-1];     //in channels
    reg [7:0]  cfg_out_ch   [0:N_LAYERS-1];     //out channels
    reg [3:0]  cfg_kH       [0:N_LAYERS-1];     //kernel height 
    reg [3:0]  cfg_kW       [0:N_LAYERS-1];     //kernel width 
    reg [1:0]  cfg_stride_h [0:N_LAYERS-1];     //stride height 
    reg [1:0]  cfg_stride_w [0:N_LAYERS-1];     //stride width 
    reg [3:0]  cfg_pad_h    [0:N_LAYERS-1];     //padding height 
    reg [3:0]  cfg_pad_w    [0:N_LAYERS-1];     //padding width 
    reg        cfg_dw       [0:N_LAYERS-1];     //depthwise = 1, pointwise = 0 
    reg [12:0] cfg_w_off    [0:N_LAYERS-1];     //weight offset for sram access 
    reg [4:0]  cfg_shift    [0:N_LAYERS-1];     //right shift amount after MAC calculation 
    reg        cfg_relu     [0:N_LAYERS-1];     //Relu 1= on, 0 = off 
    reg [7:0]  cfg_ofmap_h  [0:N_LAYERS-1];     //output feature map height 
    reg [7:0]  cfg_ofmap_w  [0:N_LAYERS-1];     //ouput feature map width 
    reg [7:0]  cfg_bias_off  [0:N_LAYERS-1];  // bias SRAM base address per layer 


    //Hard-Baked Initialization values for DFFs at each layer 
    initial begin
        //             
        cfg_in_ch[0]   =  1; cfg_out_ch[0]  = 24; cfg_kH[0]  = 10; cfg_kW[0]  = 4;
        cfg_stride_h[0]= 2; cfg_stride_w[0] =  2; cfg_pad_h[0]=  4; cfg_pad_w[0]= 1;
        cfg_dw[0]      =  0; cfg_w_off[0]   =  0; cfg_shift[0]= 14; cfg_relu[0]= 1;
        cfg_ofmap_h[0] = 25; cfg_ofmap_w[0] = 20;

        // DS block 0 depthwise
        cfg_in_ch[1]   = 24; cfg_out_ch[1]  = 24; cfg_kH[1]  =  3; cfg_kW[1]  = 3;
        cfg_stride_h[1]= 1; cfg_stride_w[1] =  1; cfg_pad_h[1]=  1; cfg_pad_w[1]= 1;
        cfg_dw[1]      =  1; cfg_w_off[1]   = 960; cfg_shift[1]= 7; cfg_relu[1]= 1;
        cfg_ofmap_h[1] = 25; cfg_ofmap_w[1] = 20;

        // DS block 0 pointwise
        cfg_in_ch[2]   = 24; cfg_out_ch[2]  = 24; cfg_kH[2]  =  1; cfg_kW[2]  = 1;
        cfg_stride_h[2]= 1; cfg_stride_w[2] =  1; cfg_pad_h[2]=  0; cfg_pad_w[2]= 0;
        cfg_dw[2]      =  0; cfg_w_off[2]   = 1176; cfg_shift[2]= 8; cfg_relu[2]= 1;
        cfg_ofmap_h[2] = 25; cfg_ofmap_w[2] = 20;

        // DS block 1 depthwise
        cfg_in_ch[3]   = 24; cfg_out_ch[3]  = 24; cfg_kH[3]  =  3; cfg_kW[3]  = 3;
        cfg_stride_h[3]= 1; cfg_stride_w[3] =  1; cfg_pad_h[3]=  1; cfg_pad_w[3]= 1;
        cfg_dw[3]      =  1; cfg_w_off[3]   = 1752; cfg_shift[3]= 7; cfg_relu[3]= 1;
        cfg_ofmap_h[3] = 25; cfg_ofmap_w[3] = 20;

        // DS block 1 pointwise
        cfg_in_ch[4]   = 24; cfg_out_ch[4]  = 24; cfg_kH[4]  =  1; cfg_kW[4]  = 1;
        cfg_stride_h[4]= 1; cfg_stride_w[4] =  1; cfg_pad_h[4]=  0; cfg_pad_w[4]= 0;
        cfg_dw[4]      =  0; cfg_w_off[4]   = 1968; cfg_shift[4]= 8; cfg_relu[4]= 1;
        cfg_ofmap_h[4] = 25; cfg_ofmap_w[4] = 20;

        // DS block 2 depthwise
        cfg_in_ch[5]   = 24; cfg_out_ch[5]  = 24; cfg_kH[5]  =  3; cfg_kW[5]  = 3;
        cfg_stride_h[5]= 1; cfg_stride_w[5] =  1; cfg_pad_h[5]=  1; cfg_pad_w[5]= 1;
        cfg_dw[5]      =  1; cfg_w_off[5]   = 2544; cfg_shift[5]= 7; cfg_relu[5]= 1;
        cfg_ofmap_h[5] = 25; cfg_ofmap_w[5] = 20;

        // DS block 2 pointwise
        cfg_in_ch[6]   = 24; cfg_out_ch[6]  = 24; cfg_kH[6]  =  1; cfg_kW[6]  = 1;
        cfg_stride_h[6]= 1; cfg_stride_w[6] =  1; cfg_pad_h[6]=  0; cfg_pad_w[6]= 0;
        cfg_dw[6]      =  0; cfg_w_off[6]   = 2760; cfg_shift[6]= 8; cfg_relu[6]= 1;
        cfg_ofmap_h[6] = 25; cfg_ofmap_w[6] = 20;

        // DS block 3 depthwise
        cfg_in_ch[7]   = 24; cfg_out_ch[7]  = 24; cfg_kH[7]  =  3; cfg_kW[7]  = 3;
        cfg_stride_h[7]= 1; cfg_stride_w[7] =  1; cfg_pad_h[7]=  1; cfg_pad_w[7]= 1;
        cfg_dw[7]      =  1; cfg_w_off[7]   = 3336; cfg_shift[7]= 7; cfg_relu[7]= 1;
        cfg_ofmap_h[7] = 25; cfg_ofmap_w[7] = 20;

        // DS block 3 pointwise
        cfg_in_ch[8]   = 24; cfg_out_ch[8]  = 24; cfg_kH[8]  =  1; cfg_kW[8]  = 1;
        cfg_stride_h[8]= 1; cfg_stride_w[8] =  1; cfg_pad_h[8]=  0; cfg_pad_w[8]= 0;
        cfg_dw[8]      =  0; cfg_w_off[8]   = 3552; cfg_shift[8]= 8; cfg_relu[8]= 1;
        cfg_ofmap_h[8] = 25; cfg_ofmap_w[8] = 20;

        // Classifier (global avg pool output is 1×1)
        cfg_in_ch[9]   = 24; cfg_out_ch[9]  =  7; cfg_kH[9]  =  1; cfg_kW[9]  = 1;
        cfg_stride_h[9]= 1; cfg_stride_w[9] =  1; cfg_pad_h[9]=  0; cfg_pad_w[9]= 0;
        cfg_dw[9]      =  0; cfg_w_off[9]   = 4128; cfg_shift[9]= 6; cfg_relu[9]= 0;
        cfg_ofmap_h[9] =  1; cfg_ofmap_w[9] =  1;
    end

    //FSM 
    localparam  IDLE        = 3'd0,
                LOAD_LAYER  = 3'd1,
                CLEAR_ACC   = 3'd2,
                COMPUTE     = 3'd3,
                WRITE_OFMAP = 3'd4,
                NEXT_PIXEL  = 3'd5,
                NEXT_LAYER  = 3'd6,
                OUTPUT      = 3'd7;

    reg [2:0]  state;
    reg [3:0]  layer;       // current layer index 0-9
    reg        buf_sel;     // 0=read A write B, 1=read B write A
    reg [7:0]  oh, ow, oc;  // output pixel coordinates
    reg [7:0]  ic;          // input channel counter
    reg [3:0]  kh, kw;      // kernel position counters
    reg [3:0]  mac_idx;     // position within N_MACS group

    // Argmax registers for classifier output
    reg signed [ACC_W-1:0] max_val;     //holds highest classification value 
    reg [2:0]              max_idx;     //Which class had the highest score [2:0] to hold 7 classes 
    reg [2:0]              oc_reg;      //register copy of oc "output channel" so we capture the correct class with highest score 

    always @(posedge clk) begin
        if (reset) begin
            state    <= IDLE;
            done     <= 0;          //when classification is done
            layer    <= 0;
            buf_sel  <= 0;
            mac_en   <= 0;
            mac_clear<= 0;
            fs_a_we  <= 0;
            fs_b_we  <= 0;
        end else begin
            // Default signal states
            mac_en    <= 0;
            mac_clear <= 0;
            fs_a_we   <= 0;
            fs_b_we   <= 0;
            done      <= 0;

            case (state)

                IDLE: begin
                    if (start) begin
                        layer   <= 0;
                        buf_sel <= 0;       //read from bank A (mel spectrogram)
                        state   <= LOAD_LAYER;
                    end
                end

                LOAD_LAYER: begin
                    // Latch current layer config into working registers    (one cycle)
                    rq_shift  <= cfg_shift[layer];
                    rq_relu_en<= cfg_relu[layer];
                    oh <= 0; ow <= 0; oc <= 0;
                    state <= CLEAR_ACC;
                end

                CLEAR_ACC: begin
                    mac_bias  <= 32'sh0;   // bias=0 (folded into BN during QAT)    (one cycle)
                    mac_clear <= 1;
                    ic <= 0; kh <= 0; kw <= 0; mac_idx <= 0;
                    state <= COMPUTE;
                end

                COMPUTE: begin
                    // Feed one (ifmap, weight) pair into MAC array per cycle
                    // Address arithmetic for weight SRAM and feature SRAM
                    // omitted here for brevity — expand per layer type
                    mac_en  <= 1;
                    // Increment kw→kh→ic counters, move to WRITE_OFMAP when done
                    if (kw == cfg_kW[layer]-1 && kh == cfg_kH[layer]-1 &&
                        ic == cfg_in_ch[layer]-1) begin
                        mac_en <= 0;
                        state  <= WRITE_OFMAP;
                    end else begin
                        // advance kernel/channel counters
                        if (kw < cfg_kW[layer]-1)       kw <= kw + 1;
                        else begin kw <= 0;
                            if (kh < cfg_kH[layer]-1)   kh <= kh + 1;
                            else begin kh <= 0;
                                if (!cfg_dw[layer])
                                    ic <= ic + 1;
                            end
                        end
                    end
                end

                WRITE_OFMAP: begin
                    // Write requantized result to output feature SRAM
                    if (!buf_sel) begin
                        fs_b_we    <= 1;
                        fs_b_waddr <= oc * cfg_ofmap_h[layer] * cfg_ofmap_w[layer]
                                    + oh * cfg_ofmap_w[layer] + ow;
                        fs_b_wdata <= rq_out;
                    end else begin
                        fs_a_we    <= 1;
                        fs_a_waddr <= oc * cfg_ofmap_h[layer] * cfg_ofmap_w[layer]
                                    + oh * cfg_ofmap_w[layer] + ow;
                        fs_a_wdata <= rq_out;
                    end

                    // Argmax tracking for classifier layer
                    if (layer == 9) begin
                        if (oc == 0 || $signed(mac_acc) > $signed(max_val)) begin
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
                    end else begin ow <= 0;
                        if (oh < cfg_ofmap_h[layer]-1) begin
                            oh    <= oh + 1;
                            state <= CLEAR_ACC;
                        end else begin oh <= 0;
                            if (oc < cfg_out_ch[layer]-1) begin
                                oc    <= oc + 1;
                                state <= CLEAR_ACC;
                            end else begin
                                oc    <= 0;
                                state <= NEXT_LAYER;
                            end
                        end
                    end
                end

                NEXT_LAYER: begin
                    buf_sel <= ~buf_sel;   // swap ping-pong buffers
                    if (layer == N_LAYERS-1)
                        state <= OUTPUT;
                    else begin
                        layer <= layer + 1;
                        state <= LOAD_LAYER;
                    end
                end

                OUTPUT: begin
                    class_out <= max_idx;
                    done      <= 1;
                    state     <= IDLE;
                end

            endcase
        end
    end

endmodule