#include <cstdint>
#include "../config.hpp"

typedef int16_t DTYPE;


void TransposeConv2d_kernel(DTYPE *in, DTYPE *bias, DTYPE *kernel, DTYPE *out);