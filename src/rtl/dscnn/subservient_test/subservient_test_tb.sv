`timescale 1ns/1ps
`default_nettype none

module subservient_test_tb;

    // Clock and reset
    reg clk  = 0;
    reg rst_n = 0;

    always #5 clk = ~clk; // 100 MHz

    initial begin
        repeat(10) @(posedge clk);
        rst_n = 1;
    end

    // Signals
    wire        ws_we;
    wire [12:0] ws_waddr;
    wire [7:0]  ws_wdata;
    reg  [12:0] ws_raddr = 0;
    wire [7:0]  ws_rdata;
    wire        weights_ready;

    // set inference always idle bc not testing with FSM yet
    wire inference_idle = 1'b1;

    // DUT: subservient wrapper
    subservient_wrapper #(
        .BOOT_MEMSIZE(512),
        .FIRMWARE("fw/firmware.hex")
    ) u_wrapper (
        .clk            (clk),
        .rst_n          (rst_n),
        .ws_we          (ws_we),
        .ws_waddr       (ws_waddr),
        .ws_wdata       (ws_wdata),
        .inference_idle (inference_idle),
        .weights_ready  (weights_ready)
    );

    // Weight SRAM
    weight_sram u_weight_sram (
        .clk   (clk),
        .we    (ws_we),
        .waddr (ws_waddr),
        .wdata (ws_wdata),
        .raddr (ws_raddr),
        .rdata (ws_rdata)
    );

    // Trace every weight SRAM write
    always @(posedge clk)
        if (ws_we)
            $display("[%0t ns] WRITE weight_sram[%0d] = 8'h%02h",
                     $time, ws_waddr, ws_wdata);

    // Debug Wishbone bus
    always @(posedge clk) begin
        if (u_wrapper.wb_stb)
            $display("[%0t ns] WB: stb=%b we=%b adr=%08h dat=%08h ack=%b",
                     $time,
                     u_wrapper.wb_stb,
                     u_wrapper.wb_we,
                     u_wrapper.wb_adr,
                     u_wrapper.wb_dat,
                     u_wrapper.wb_ack);
    end

    // Debug boot SRAM reads to confirm SERV is fetching
    always @(posedge clk) begin
        if (u_wrapper.proc_sram_ren)
            $display("[%0t ns] SRAM read: addr=%0d data=%02h",
                     $time,
                     u_wrapper.proc_sram_raddr,
                     u_wrapper.proc_sram_rdata);
    end

    // main test
    integer i;
    integer pass;

    initial begin
        $dumpfile("sim.vcd");
        $dumpvars(0, subservient_test_tb);

        $display("Waiting for weights_ready...");

        pass = 0;
        for (i = 0; i < 10_000_000; i = i + 1) begin
            @(posedge clk);
            if (weights_ready) begin
                pass = 1;
                i = 10_000_000;
            end
            if (i % 500_000 == 0)
                $display("[%0d cycles] still waiting...", i);
        end

        if (!pass) begin
            $display("TIMEOUT — weights_ready never asserted");
            $finish;
        end

        $display("weights_ready asserted at %0t ns", $time);

        // Read back first 8 weight addresses (1-cycle read latency)
        for (i = 0; i < 8; i = i + 1) begin
            ws_raddr = i[12:0];
            @(posedge clk);
            @(posedge clk);
            $display("weight_sram[%0d] = 8'h%02h", i, ws_rdata);
        end

        $display("PASS");
        $finish;
    end

endmodule

`default_nettype wire