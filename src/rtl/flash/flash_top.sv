// flash_top.sv — Used to test verify flash FSM and Routers
// using simple reg_arrays as placeholders for SRAM Macros.
//
// This is a SIMULATION-ONLY harness. Real silicon integration lives
// in chip_core.sv, which instantiates the boot subsystem directly.

module flash_top
    import boot_pkg::*;
(
    input  logic        clk,
    input  logic        reset,

    //UART byte interface
    input  logic [7:0]  rx_byte_i,
    input  logic        rx_valid_i,
    output logic        rx_ready_o,

    output logic [7:0]  tx_byte_o,
    output logic        tx_valid_o,
    input  logic        tx_ready_i,

    // Status
    output logic        boot_done_o,
    output logic        pkt_valid_o,

    // Verification read-back ports
    input  logic [5:0]  verify_lut_addr_i,
    output logic [15:0] verify_lut_data_o,

    // Weight memory read-back
    input  logic [12:0] verify_weight_addr_i,
    output logic [7:0]  verify_weight_data_o,

    // Layer config read-back
    input  logic [7:0]  verify_cfg_addr_i,
    output logic [7:0]  verify_cfg_data_o,

    // Mel coefficient read-back
    input  logic [7:0]  verify_mel_addr_i,
    output logic [15:0] verify_mel_data_o,

    // Mel meta read-back
    input  logic [7:0]  verify_meta_addr_i,
    output logic [7:0]  verify_meta_data_o
);

    //  Internal wires

    // Boot buses
    boot_bus_t features_boot;
    boot_bus_t dscnn_boot;

    // Features router → log_lut
    logic        lut_boot_we;
    logic [5:0]  lut_boot_addr;
    logic [15:0] lut_boot_wdata;

    // Features router → mel_coeff_sram
    logic        mel_boot_we;
    logic [7:0]  mel_boot_addr;
    logic [15:0] mel_boot_wdata;

    // Features router → mel meta (start/end/offset)
    logic        meta_boot_we;
    logic [7:0]  meta_boot_addr;
    logic [7:0]  meta_boot_wdata;

    // DS-CNN router → weight memory
    logic        w_boot_we;
    logic [12:0] w_boot_addr;
    logic [7:0]  w_boot_wdata;

    // DS-CNN router → layer config
    logic        cfg_boot_we;
    logic [7:0]  cfg_boot_addr;
    logic [7:0]  cfg_boot_wdata;

    //  Boot controller
    boot_controller u_boot_ctrl (
        .clk_i           (clk),
        .reset_i         (reset),

        .rx_byte_i       (rx_byte_i),
        .rx_valid_i      (rx_valid_i),
        .rx_ready_o      (rx_ready_o),

        .tx_byte_o       (tx_byte_o),
        .tx_valid_o      (tx_valid_o),
        .tx_ready_i      (tx_ready_i),

        .features_boot_o (features_boot),
        .dscnn_boot_o    (dscnn_boot),

        .boot_done_o     (boot_done_o),
        .pkt_valid_o     (pkt_valid_o),
        .last_target_o   (),
        .last_addr_o     (),
        .last_len_o      ()
    );

    //  Features boot router
    features_boot_router u_feat_router (
        .boot_i            (features_boot),

        .lut_boot_we_o     (lut_boot_we),
        .lut_boot_addr_o   (lut_boot_addr),
        .lut_boot_wdata_o  (lut_boot_wdata),

        .mel_boot_we_o     (mel_boot_we),
        .mel_boot_addr_o   (mel_boot_addr),
        .mel_boot_wdata_o  (mel_boot_wdata),

        .meta_boot_we_o    (meta_boot_we),
        .meta_boot_addr_o  (meta_boot_addr),
        .meta_boot_wdata_o (meta_boot_wdata)
    );

    //  DS-CNN boot router
    dscnn_boot_router u_dscnn_router (
        .boot_i             (dscnn_boot),

        .w_boot_we_o        (w_boot_we),
        .w_boot_addr_o      (w_boot_addr),
        .w_boot_wdata_o     (w_boot_wdata),

        .cfg_boot_we_o      (cfg_boot_we),
        .cfg_boot_addr_o    (cfg_boot_addr),
        .cfg_boot_wdata_o   (cfg_boot_wdata)
    );

    //  Log LUT — behavioral stand-in for test
    reg [15:0] lut_mem [0:63];

    always_ff @(posedge clk) begin
        if (lut_boot_we)
            lut_mem[lut_boot_addr] <= lut_boot_wdata;
    end

    reg [15:0] verify_lut_data_q;
    always_ff @(posedge clk)
        verify_lut_data_q <= lut_mem[verify_lut_addr_i];
    assign verify_lut_data_o = verify_lut_data_q;


    //  Mel coefficient memory — behavioral stand-in for test (256 × 16-bit sparse)
    reg [15:0] mel_mem [0:255];

    always_ff @(posedge clk) begin
        if (mel_boot_we)
            mel_mem[mel_boot_addr] <= mel_boot_wdata;
    end

    reg [15:0] verify_mel_data_q;
    always_ff @(posedge clk)
        verify_mel_data_q <= mel_mem[verify_mel_addr_i];
    assign verify_mel_data_o = verify_mel_data_q;


    //  Mel metadata — behavioral stand-in for test (256 × 8-bit)
    reg [7:0] meta_mem [0:255];

    always_ff @(posedge clk) begin
        if (meta_boot_we)
            meta_mem[meta_boot_addr] <= meta_boot_wdata;
    end

    reg [7:0] verify_meta_data_q;
    always_ff @(posedge clk)
        verify_meta_data_q <= meta_mem[verify_meta_addr_i];
    assign verify_meta_data_o = verify_meta_data_q;


    //  Weight memory — behavioral stand-in for test (8192 × 8-bit, sized to router addr)
    reg [7:0] weight_mem [0:8191];

    always_ff @(posedge clk) begin
        if (w_boot_we)
            weight_mem[w_boot_addr] <= w_boot_wdata;
    end

    reg [7:0] verify_weight_data_q;
    always_ff @(posedge clk)
        verify_weight_data_q <= weight_mem[verify_weight_addr_i];
    assign verify_weight_data_o = verify_weight_data_q;


    //  Layer config — behavioral stand-in for test
    reg [7:0] cfg_mem [0:255];

    always_ff @(posedge clk) begin
        if (cfg_boot_we)
            cfg_mem[cfg_boot_addr] <= cfg_boot_wdata;
    end

    reg [7:0] verify_cfg_data_q;
    always_ff @(posedge clk)
        verify_cfg_data_q <= cfg_mem[verify_cfg_addr_i];
    assign verify_cfg_data_o = verify_cfg_data_q;

endmodule
