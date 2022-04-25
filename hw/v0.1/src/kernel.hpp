#include "../../config.hpp"
#include "ap_int.h"

typedef ap_int<16> HWTYPE;

extern "C" {
void TransposeConv2d_kernel(HWTYPE *in, HWTYPE *bias, HWTYPE *kernel, HWTYPE *out);
}
