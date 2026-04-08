#include <stdint.h>

#define WEIGHT_BASE  0x00000000
#define READY_REG    0x00002000

void main(void) {
    // Write test pattern
    volatile uint8_t *sram = (volatile uint8_t *)WEIGHT_BASE;
    for (int i = 0; i < 256; i++)
        sram[i] = (uint8_t)(i & 0xFF);

    // signal done
    *(volatile uint32_t *)READY_REG = 1;

    while (1) asm volatile("nop");
}