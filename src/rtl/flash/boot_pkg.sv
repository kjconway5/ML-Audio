// boot_pkg.sv
// Shared definitions for the boot/flash subsystem.

package boot_pkg;

    //  Bus widths:
    // BOOT_ADDR_W is sized for the largest target SRAM:
    // BOOT_DATA_W is 16 because the widest packed target is 16-bit
    localparam BOOT_ADDR_W   = 16;
    localparam BOOT_DATA_W   = 16;
    localparam BOOT_TARGET_W = 8;
    localparam MODULE_SEL_W  = 4;
    localparam SUBTARGET_W   = 4;

    //  Top-level module encoding (high nibble of target byte)
    typedef enum logic [MODULE_SEL_W-1:0] {
        MOD_FEATURES = 4'h0,   // Target SRAMS in different modules found within Features Pipeline
        MOD_DSCNN    = 4'h1,   // weights, layer config
        MOD_DEBUG    = 4'h8,   // DFT
        MOD_CONTROL  = 4'hF    // Booting (maybe additional run-time operations)
    } module_sel_e;


    //  Features pipeline subtargets (low nibble when MOD_FEATURES)
    // Hann window and FFT twiddles are hard ROMs ($readmemh in
    // stfft.sv / fftstage.v) and are NOT flash-loadable.
    typedef enum logic [SUBTARGET_W-1:0] {
        FEAT_LOG_LUT   = 4'h0,   // 64  × 16-bit  → 2× sram256x8
        FEAT_MEL_COEFF = 4'h1,   // 256 × 16-bit  → 2× sram256x8 (sparse packing)
        FEAT_MEL_META  = 4'h2    // 256 × 8-bit   → 1× sram256x8 (starts/ends/offsets)
    } features_subtarget_e;


    // DS-CNN subtargets (low nibble when MOD_DSCNN)
    typedef enum logic [SUBTARGET_W-1:0] {
        DSCNN_WEIGHTS = 4'h0,   // 4296 × 8-bit  → 9× sram512x8
        DSCNN_CFG     = 4'h1    // layer config registers
    } dscnn_subtarget_e;


    //  Control subtargets (low nibble when MOD_CONTROL)
    typedef enum logic [SUBTARGET_W-1:0] {
        CTRL_BOOT_DONE = 4'h0,   // assert boot_done, release inference reset
        CTRL_START     = 4'h1    // trigger an inference run (Possible move to MOD_DEBUG)
    } control_subtarget_e;



    //  UART protocol constants

    localparam logic [7:0] SYNC_BYTE_0 = 8'hAA;   // first sync byte
    localparam logic [7:0] SYNC_BYTE_1 = 8'h55;   // second sync byte

    localparam logic [7:0] ACK_BYTE    = 8'h06;   // packet OK
    localparam logic [7:0] NACK_BYTE   = 8'hEE;   // checksum failed
    localparam logic [7:0] ERR_BYTE    = 8'hE1;   // malformed packet


    // Returns 1 if the target is 16-bit (2 payload bytes per SRAM word),
    // 0 if the target is 8-bit (1 payload byte per SRAM word).
    //
    // Used by boot_controller to know when to pack two bytes into one
    // bus write vs passing a single byte through.
 
    function automatic logic is_target_16bit(input logic [BOOT_TARGET_W-1:0] target);
        logic [MODULE_SEL_W-1:0]  mod;
        logic [SUBTARGET_W-1:0]   sub;
        mod = target[7:4];
        sub = target[3:0];
 
        case (mod)
            MOD_FEATURES:
                case (sub)
                    FEAT_LOG_LUT:   return 1'b1;   // 64  × 16-bit
                    FEAT_MEL_COEFF: return 1'b1;   // 256 × 16-bit
                    FEAT_MEL_META:  return 1'b0;   // 256 × 8-bit
                    default:        return 1'b0;
                endcase
            MOD_DSCNN:
                case (sub)
                    DSCNN_WEIGHTS:  return 1'b0;   // 6752 × 8-bit
                    DSCNN_CFG:      return 1'b0;   // 8-bit config registers
                    default:        return 1'b0;
                endcase
            default: return 1'b0;   // MOD_CONTROL, MOD_DEBUG: no packing
        endcase
    endfunction
    
    //  Inter-module boot bus (top-level controller -> sub-module router)
    //
    // One transaction = one packed write into a sub-module's chosen SRAM.
    //   valid     pulses high for 1 cycle when fields are valid
    //   subtarget selects which SRAM within the sub-module
    //   addr      address within the chosen SRAM
    //   data      16-bit packed write data (8-bit targets use [7:0])

    typedef struct packed {
        logic                    valid;
        logic [SUBTARGET_W-1:0]  subtarget;
        logic [BOOT_ADDR_W-1:0]  addr;
        logic [BOOT_DATA_W-1:0]  data;
    } boot_bus_t;

endpackage