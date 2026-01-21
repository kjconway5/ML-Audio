
// Run with: make test-sv

`timescale 1ns/1ps

module tb_example;

    // Parameters
    parameter WIDTH = 8;
    parameter CLK_PERIOD = 10;  // 10ns = 100MHz

    // Signals
    logic             clk_i;
    logic             rst_i;
    logic             en_i;
    logic [WIDTH-1:0] count_o;

    // Instantiate DUT
    counter #(
        .width_p(WIDTH)
    ) dut (
        .clk_i   (clk_i),
        .reset_i   (rst_i),
        .en_i    (en_i),
        .count_o (count_o)
    );

    // Clock generation
    initial begin
        clk_i = 0;
        forever #(CLK_PERIOD/2) clk_i = ~clk_i;
    end

    // Waveform dumping
    initial begin
        $dumpfile("tb_counter.vcd");
        $dumpvars(0, tb_counter);
    end

    // Test sequence
    initial begin
        $display("=== Counter Testbench Starting ===");
        
        // Initialize
        rst_i = 1;
        en_i  = 0;
        
        // Wait for a few clocks
        repeat(2) @(posedge clk_i);
        
        // Release reset
        @(posedge clk_i);
        rst_i = 0;
        
        // Enable counter
        @(posedge clk_i);
        en_i = 1;
        
        // Let it count for a while
        repeat(20) begin
            @(posedge clk_i);
            $display("Time=%0t  Count=%0d", $time, count_o);
        end
        
        // Disable counter
        en_i = 0;
        repeat(5) @(posedge clk_i);
        $display("Counter disabled - should hold value: %0d", count_o);
        
        // Re-enable
        en_i = 1;
        repeat(10) @(posedge clk_i);
        
        // Test reset during counting
        $display("Applying reset...");
        rst_i = 1;
        @(posedge clk_i);
        rst_i = 0;
        
        if (count_o == 0) begin
            $display("✓ PASS: Reset works correctly");
        end else begin
            $display("✗ FAIL: Counter not reset to 0");
        end
        
        // Continue counting
        repeat(10) @(posedge clk_i);
        
        $display("=== Counter Testbench Complete ===");
        $finish;
    end

    // Timeout watchdog
    initial begin
        #10000;
        $display("ERROR: Testbench timeout!");
        $finish;
    end

endmodule