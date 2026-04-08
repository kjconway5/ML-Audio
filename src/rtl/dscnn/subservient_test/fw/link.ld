ENTRY(_start)

MEMORY
{
  RAM (rwx) : ORIGIN = 0x00000000, LENGTH = 512
}

SECTIONS
{
  . = 0x00000000;

  .text :
  {
    *(.text.start)
    *(.text)
    *(.rodata)
    *(.data)
    *(.bss)
    *(COMMON)
  }

  /DISCARD/ :
  {
    *(.interp)
    *(.dynamic)
    *(.dynsym)
    *(.dynstr)
    *(.hash)
    *(.gnu.hash)
    *(.gnu.version*)
    *(.rel*)
    *(.note*)
    *(.eh_frame*)
  }
}