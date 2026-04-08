CROSS_COMPILE ?= riscv64-linux-gnu-
CC      := $(CROSS_COMPILE)gcc
OBJCOPY := $(CROSS_COMPILE)objcopy

CFLAGS  := -march=rv32i -mabi=ilp32 -nostdlib -nostartfiles -ffreestanding -static -Wall
LDFLAGS := -march=rv32i -mabi=ilp32 -T link.ld -nostdlib -nostartfiles -static

SOURCES := startup.S firmware.c
OBJECTS := startup.o firmware.o
ELF     := firmware.elf
HEX     := firmware.hex

.PHONY: all clean

all: $(HEX)

$(ELF): $(OBJECTS)
	$(CC) $(LDFLAGS) -o $@ $^

$(HEX): $(ELF)
	$(OBJCOPY) -O verilog --verilog-data-width=1 $< $@

%.o: %.S
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJECTS) $(ELF) $(HEX)