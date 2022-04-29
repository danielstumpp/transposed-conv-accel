# transposed-conv-accel
FPGA-based acceleration of transposed convolution


## versions

### v0.0
- Input padding + convolution 

### v0.1
- Removal of input buffer

### v0.2
- removal of input buffer
- tiling output channels and input/output dim

### v1.0 
- unrolling in by 256 in input channel dim

### v2.0
- unrolling in by 256 in output channel dim
- Multiple AXI ports
- 512-bit for input and kernel
- 256-bit for output and bias (due to channel tile size of 16)

### v2.1
- unrolling in by 256 in input channel dim
- Multiple AXI ports
- Increase tile size in output channel to 32
- 512-bit for all ports

### v2.2
- all v2.1 optimizations
- additional unrolling by one kernel dim

### v3.0
- input 512-bit stream
- bias 16-bit stream
- kernel 512-bit stream
- output 256-bit stream

### v3.1
- output 512-bit stream
- output channel tile size 32
- conditional temp assignment within compute loop

### v3.2
- deeper streams
- bias 512-bit stream
- unrolling in kernel dimension
