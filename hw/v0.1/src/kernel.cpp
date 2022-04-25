#include "kernel.hpp"

#define MAX(A, B) ((A >= B) ? A : B)

/*
void TransposeConv2d_kernel(DTYPE in[CFG::in_channels][CFG::in_size][CFG::in_size],
                     DTYPE bias[CFG::out_channels],
                     DTYPE kernel[CFG::out_channels][CFG::in_channels][CFG::kernel_size][CFG::kernel_size],
                     DTYPE out[CFG::out_channels][CFG::out_size][CFG::out_size])
                     */

extern "C" {

// v0: access buffers directly from DRAM as 1-D array
void TransposeConv2d_kernel(HWTYPE *in, HWTYPE *bias, HWTYPE *kernel, HWTYPE *out)
{
#pragma HLS interface m_axi port = in bundle = gmem0
#pragma HLS interface m_axi port = bias bundle = gmem0
#pragma HLS interface m_axi port = kernel bundle = gmem1
#pragma HLS interface m_axi port = out bundle = gmem1
#pragma HLS interface s_axilite port = in bundle = control
#pragma HLS interface s_axilite port = bias bundle = control
#pragma HLS interface s_axilite port = kernel bundle = control
#pragma HLS interface s_axilite port = out bundle = control
#pragma HLS interface s_axilite port = return bundle = control

    for (int i = 0; i < CFG::out_channels; ++i){
        for (int h = 0; h < CFG::out_size; ++h){
            for (int w = 0; w < CFG::out_size; ++w){
                out[i*CFG::out_size*CFG::out_size + h*CFG::out_size + w] = bias[i];
            }
        }
    }

    // This first part is where most of the weird stuff happens
    const int inpad = MAX(CFG::kernel_size - CFG::pad - 1, 0);                           

    for (int i = 0; i < CFG::out_channels; ++i){
        for (int j = 0; j < CFG::in_channels; ++j){
            for (int h = 0; h < CFG::out_size; ++h){
                for (int w = 0; w < CFG::out_size; ++w) {
                    for (int p = 0; p < CFG::kernel_size; ++p) {
                        for (int q = 0; q < CFG::kernel_size; ++q)
                        {
                            if ((h + p) % CFG::stride == inpad && (w+q) % CFG::stride == inpad){
                               HWTYPE val = in[j * CFG::in_size * CFG::in_size + (h/CFG::stride) * CFG::in_size + (w/CFG::stride)];

                                out[i * CFG::out_size * CFG::out_size + h * CFG::out_size + w] +=
                                    kernel[(i * CFG::in_channels * CFG::kernel_size * CFG::kernel_size) + (j * CFG::kernel_size * CFG::kernel_size) + (p * CFG::kernel_size) + q] * val;
                            }                           
                        }
                    }
                }
            }
        }
    }

}

}
