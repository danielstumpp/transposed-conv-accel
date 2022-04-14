#include <cstdint>
#include "../config.hpp"

typedef int16_t DTYPE;


void TransposeConv2d(DTYPE in[CFG::in_channels][CFG::in_size][CFG::in_size],
                     DTYPE bias[CFG::out_channels],
                     DTYPE kernel[CFG::out_channels][CFG::in_channels][CFG::kernel_size][CFG::kernel_size],
                     DTYPE out[CFG::out_channels][CFG::out_size][CFG::out_size]);

void TransposeConv2d_arr(DTYPE *in,
                     DTYPE *bias,
                     DTYPE *kernel,
                     DTYPE *out);
