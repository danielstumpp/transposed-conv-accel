src_files = tconv_tb.cpp TransposeConv2d.cpp

.PHONY: all

all: tb

tb: $(src_files) 
	g++ -Wall --std=c++14 -o tb $(src_files)

.PHONY: clean

clean:
	rm tb