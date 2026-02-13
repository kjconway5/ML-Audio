// lau_pkg.sv
package lau_pkg;
    typedef enum logic [1:0] {
        SLOW   = 2'd0,
        MEDIUM = 2'd1,
        FAST   = 2'd2
    } speed_e;
endpackage
