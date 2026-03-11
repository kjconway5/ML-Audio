BITSTREAM := demo.bit
MODEL     := demo.pt
PORT      := /dev/ttyUSB1
BOARD     := arty_a7_100t

flash:
	openFPGALoader -b $(BOARD) "$(BITSTREAM)"

demo: flash
	python3 stream_demo2.py -p $(PORT) -m $(MODEL)
