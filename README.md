# transposed-conv-accel
FPGA-based acceleration of transposed convolution


## versions

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
