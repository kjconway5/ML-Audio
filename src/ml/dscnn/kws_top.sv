// kws_top.v
// Top-level module — wires all submodules together
// This is the only file Yosys needs as the top for synthesis

module kws_top (
    input  wire        clk,
    input  wire        reset,
    input  wire        start,
    output wire        done,
    output wire [2:0]  class_out
);
    // Internal wires
    wire [12:0]          w_addr;
    wire signed [7:0]    w_data;

    wire                 fs_a_we;
    wire [13:0]          fs_a_waddr, fs_a_raddr;
    wire signed [7:0]    fs_a_wdata, fs_a_rdata;

    wire                 fs_b_we;
    wire [13:0]          fs_b_waddr, fs_b_raddr;
    wire signed [7:0]    fs_b_wdata, fs_b_rdata;

    wire                 mac_en, mac_clear;
    wire signed [7:0]    mac_ifmap  [0:15];
    wire signed [7:0]    mac_weight [0:15];
    wire signed [31:0]   mac_bias, mac_acc;
    wire                 mac_valid;

    wire [4:0]           rq_shift;
    wire                 rq_relu_en;
    wire signed [7:0]    rq_out;

    // Instantiate submodules
    weight_sram  u_wsram  (.clk(clk), .addr(w_addr),   .data(w_data));

    feature_sram u_fsram  (
        .clk(clk),
        .a_we(fs_a_we), .a_waddr(fs_a_waddr), .a_wdata(fs_a_wdata),
        .a_raddr(fs_a_raddr), .a_rdata(fs_a_rdata),
        .b_we(fs_b_we), .b_waddr(fs_b_waddr), .b_wdata(fs_b_wdata),
        .b_raddr(fs_b_raddr), .b_rdata(fs_b_rdata)
    );

    mac_array    u_mac    (
        .clk(clk), .reset(reset),
        .en(mac_en), .clear(mac_clear),
        .ifmap(mac_ifmap), .weight(mac_weight),
        .bias(mac_bias), .acc(mac_acc), .valid(mac_valid)
    );

    requant      u_rq     (
        .acc(mac_acc), .shift(rq_shift),
        .relu_en(rq_relu_en), .out(rq_out)
    );

    layer_controller u_ctrl (
        .clk(clk), .reset(reset), .start(start),
        .done(done), .class_out(class_out),
        .w_addr(w_addr), .w_data(w_data),
        .fs_a_we(fs_a_we), .fs_a_waddr(fs_a_waddr), .fs_a_wdata(fs_a_wdata),
        .fs_a_raddr(fs_a_raddr), .fs_a_rdata(fs_a_rdata),
        .fs_b_we(fs_b_we), .fs_b_waddr(fs_b_waddr), .fs_b_wdata(fs_b_wdata),
        .fs_b_raddr(fs_b_raddr), .fs_b_rdata(fs_b_rdata),
        .mac_en(mac_en), .mac_clear(mac_clear),
        .mac_ifmap(mac_ifmap), .mac_weight(mac_weight),
        .mac_bias(mac_bias), .mac_acc(mac_acc), .mac_valid(mac_valid),
        .rq_shift(rq_shift), .rq_relu_en(rq_relu_en), .rq_out(rq_out)
    );

endmodule
```

---

The one thing marked as incomplete in `layer_controller.v` is the SRAM address arithmetic inside `COMPUTE` — that's the trickiest part and deserves its own focused pass once the rest compiles cleanly. The address formula for reading ifmap is:
```
ifmap_addr = ic * ifmap_H * ifmap_W
           + (oh * stride_h + kh) * ifmap_W
           + (ow * stride_w + kw)
```

And for weights:
```
// Standard conv:   weight_addr = w_offset + oc*(in_ch*kH*kW) + ic*(kH*kW) + kh*kW + kw
// Depthwise conv:  weight_addr = w_offset + oc*(kH*kW) + kh*kW + kw