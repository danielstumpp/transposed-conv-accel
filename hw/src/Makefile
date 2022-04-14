src_files = ./kernel.cpp ./host.cpp
header_files = ../config.hpp ./kernel.hpp

.PHONY: all

all: tb

tb: $(src_files) $(header_files)
	g++ -Wall --std=c++14 -o host $(src_files)

.PHONY: clean

clean:
	rm host