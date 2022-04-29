#include "../../config.hpp"
#include "ap_int.h"
#include <stdint.h>

#define MAX(A, B) ((A >= B) ? A : B)
#define MIN(A, B) ((A >= B) ? B : A)

typedef int16_t HWTYPE;
typedef ap_int<512> block512_t;
typedef ap_int<256> block256_t;
const int WIDTH512 = 32; 
const int WIDTH256 = 16;
const int WORD_BITS = 16;

extern "C" {
void TransposeConv2d_kernel(block512_t *in, block512_t *bias, block512_t *kernel, block512_t *out);
}
