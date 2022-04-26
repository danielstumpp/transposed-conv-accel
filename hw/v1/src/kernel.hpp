#include "../../config.hpp"
#include "ap_int.h"

#define MAX(A, B) ((A >= B) ? A : B)
#define MIN(A, B) ((A >= B) ? B : A)

typedef ap_int<16> HWTYPE;

extern "C" {
void TransposeConv2d_kernel(HWTYPE *in, HWTYPE *bias, HWTYPE *kernel, HWTYPE *out);
}
