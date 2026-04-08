.section .text.start
.global _start

_start:
    li sp, 0x1F0  # top of RAM minus 16 bytes safety
    call main
1:  j 1b