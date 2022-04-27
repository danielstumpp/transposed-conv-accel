#include "../../config.hpp"
#include "ap_int.h"

#define MAX(A, B) ((A >= B) ? A : B)
#define MIN(A, B) ((A >= B) ? B : A)

typedef ap_int<16> HWTYPE;
typedef ap_int<400> os_block_t;
const int OS_BW = 25;  // (output size block width) 25 HWTYPE per os_block_t
const int WORD_BITS = 16; 

extern "C" {
void TransposeConv2d_kernel(HWTYPE *in, HWTYPE *bias, HWTYPE *kernel, HWTYPE *out);
}
