`default_nettype none

module subservient_wrapper #(
    parameter BOOT_MEMSIZE = 512,
    parameter FIRMWARE     = "fw/firmware.hex"
)(
    input  wire        clk,
    input  wire        rst_n,

    // Weight SRAM write port
    output wire        ws_we,
    output wire [12:0] ws_waddr,
    output wire [7:0]  ws_wdata,

    // Handshake
    input  wire        inference_idle,
    output reg         weights_ready
);

    initial begin
        // Initialize boot memory to zeroes
        integer j;
        for (j = 0; j < BOOT_MEMSIZE; j = j + 1)
            boot_mem[j] = 8'h00;
        $readmemh(FIRMWARE, boot_mem);
        $display("Loaded firmware from %s", FIRMWARE);
    end

    localparam AW = $clog2(BOOT_MEMSIZE);

    // Boot SRAM
`ifdef SIM
    reg [7:0] boot_mem [0:BOOT_MEMSIZE-1];

    initial begin
        $readmemh(FIRMWARE, boot_mem);
        $display("Loaded firmware from %s", FIRMWARE);
    end

    wire [AW-1:0] proc_sram_waddr;
    wire [7:0]    proc_sram_wdata;
    wire          proc_sram_wen;
    wire [AW-1:0] proc_sram_raddr;
    wire [7:0]    proc_sram_rdata;
    wire          proc_sram_ren;

    reg [7:0] boot_mem_rdata;

    always @(posedge clk)
        if (proc_sram_ren)
            boot_mem_rdata <= boot_mem[proc_sram_raddr];

    always @(posedge clk)
        if (proc_sram_wen)
            boot_mem[proc_sram_waddr] <= proc_sram_wdata;

    assign proc_sram_rdata = boot_mem_rdata;

`else
    // for synthesis infer actual SRAM
    wire [AW-1:0] proc_sram_waddr;
    wire [7:0]    proc_sram_wdata;
    wire          proc_sram_wen;
    wire [AW-1:0] proc_sram_raddr;
    wire [7:0]    proc_sram_rdata;
    wire          proc_sram_ren;

    gf180mcu_fd_ip_sram__sram512x8m8wm1 u_insn_sram (
        .CLK  (clk),
        .CEN  (~(proc_sram_wen | proc_sram_ren)),
        .GWEN (~proc_sram_wen),
        .WEN  ({8{~proc_sram_wen}}),
        .A    (proc_sram_wen ? proc_sram_waddr : proc_sram_raddr),
        .D    (proc_sram_wdata),
        .Q    (proc_sram_rdata)
    );
`endif

    // Wishbone peripheral interface wires
    wire [31:0] wb_adr;
    wire [31:0] wb_dat;
    wire [3:0]  wb_sel;
    wire        wb_we;
    wire        wb_stb;
    wire [31:0] wb_rdt;
    wire        wb_ack;

    // Subservient core
    subservient_core #(
        .memsize (BOOT_MEMSIZE),
        .WITH_CSR(1)
    ) u_subservient (
        .i_clk       (clk),
        .i_rst       (~rst_n),
        .i_timer_irq (1'b0),

        .o_sram_waddr (proc_sram_waddr),
        .o_sram_wdata (proc_sram_wdata),
        .o_sram_wen   (proc_sram_wen),
        .o_sram_raddr (proc_sram_raddr),
        .i_sram_rdata (proc_sram_rdata),
        .o_sram_ren   (proc_sram_ren),

        .o_wb_adr (wb_adr),
        .o_wb_dat (wb_dat),
        .o_wb_sel (wb_sel),
        .o_wb_we  (wb_we),
        .o_wb_stb (wb_stb),
        .i_wb_rdt (wb_rdt),
        .i_wb_ack (wb_ack)
    );

    // Address decode
    // 0x00000000–0x00001FFF : weight SRAM (13-bit byte address)
    // 0x00002000            : ready register (write 1 to assert weights_ready)

    localparam READY_ADDR = 32'h00002000;

    wire wb_write       = wb_stb & wb_we;
    wire sram_region    = (wb_adr[31:13] == 19'd0);    // addr < 0x2000
    wire ready_region   = (wb_adr == READY_ADDR);

    // Gate weight writes on inference_idle, stalls Wishbone if FSM is busy
    // Subservient will spin waiting for wb_ack, which only comes when idle
    wire safe_to_write  = wb_write & inference_idle;

    // Weight SRAM drive
    assign ws_we    = safe_to_write & sram_region & wb_sel[0];
    assign ws_waddr = wb_adr[12:0];
    assign ws_wdata = wb_dat[7:0];

    // weights_ready register
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            weights_ready <= 1'b0;
        else if (safe_to_write && ready_region && wb_sel[0])
            weights_ready <= wb_dat[0];
    end

    // Wishbone ack: registered, one cycle after stb
    // Held low if inference_idle is blocking a weight write
    reg wb_ack_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            wb_ack_r <= 1'b0;
        else
            // Ack any transaction except a weight write that's being stalled
            wb_ack_r <= wb_stb & (ready_region | sram_region & inference_idle);
    end

    assign wb_ack = wb_ack_r;
    assign wb_rdt = 32'd0;

endmodule

`default_nettype wire