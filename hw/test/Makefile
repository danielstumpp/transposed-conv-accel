src_files = ./tconv_tb.cpp ../src/TransposeConv2d.cpp ../src/kernel.cpp
header_files = ../config.hpp ../src/TransposeConv2d.hpp ../src/kernel.hpp

.PHONY: all

all: tb

tb: $(src_files) $(header_files)
	g++ -Wall --std=c++14 -o tb $(src_files)

.PHONY: clean

clean:
	rm tb