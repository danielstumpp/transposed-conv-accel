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
#pragma HLS interface m_axi port = in offset = slave bundle = gmem
#pragma HLS interface m_axi port = bias offset = slave bundle = gmem
#pragma HLS interface m_axi port = kernel offset = slave bundle = gmem
#pragma HLS interface m_axi port = out offset = slave bundle = gmem
#pragma HLS interface s_axilite port = in bundle = control
#pragma HLS interface s_axilite port = bias bundle = control
#pragma HLS interface s_axilite port = kernel bundle = control
#pragma HLS interface s_axilite port = out bundle = control

    // This first part is where most of the weird stuff happens
    const int inpad = MAX(CFG::kernel_size - CFG::pad - 1, 0);                           
    const int pin_size = CFG::in_size + (CFG::in_size-1)*(CFG::stride - 1) +  2*inpad;  
    
    HWTYPE padded_in[CFG::in_channels][pin_size][pin_size];
    HWTYPE out_buf[CFG::out_channels][CFG::out_size][CFG::out_size];

    for (int c = 0; c < CFG::in_channels; ++c){
        for (int i = 0; i < pin_size; ++i){
            for (int j = 0; j < pin_size; ++j){
                padded_in[c][i][j] = 0;
            }
        }
    }
    
    for (int c = 0; c < CFG::in_channels; ++c){
        for (int h = 0; h < CFG::in_size; ++h){
            for (int w = 0; w < CFG::in_size; ++w){
                padded_in[c][inpad + h*CFG::stride][inpad + w*CFG::stride] = in[c*CFG::in_size*CFG::in_size + h*CFG::in_size + w]; 
            }
        }
    }

    for (int i = 0; i < CFG::out_channels; ++i){
        for (int h = 0; h < CFG::out_size; ++h){
            for (int w = 0; w < CFG::out_size; ++w){
                out[i*CFG::out_size*CFG::out_size + h*CFG::out_size + w] = bias[i];
            }
        }
    }

    for (int i = 0; i < CFG::out_channels; ++i){
        for (int j = 0; j < CFG::in_channels; ++j){
            for (int h = 0; h < CFG::out_size; ++h){
                for (int w = 0; w < CFG::out_size; ++w) {
                    for (int p = 0; p < CFG::kernel_size; ++p) {
                        for (int q = 0; q < CFG::kernel_size; ++q)
                        {
                            //out[i*CFG::out_size*CFG::out_size + h*CFG::out_size + w] += 
                            out_buf[i][h][w] +=
                                kernel[(i*CFG::in_channels*CFG::kernel_size*CFG::kernel_size)+(j*CFG::kernel_size*CFG::kernel_size)+(p*CFG::kernel_size) + q] 
                                    * padded_in[j][h + p][w + q];
                        }
                    }
                }
            }
        }
    }

    for (int i = 0; i < CFG::out_channels; ++i){
        for (int h = 0; h < CFG::out_size; ++h){
            for (int w = 0; w < CFG::out_size; ++w) {
                out[i*CFG::out_size*CFG::out_size + h*CFG::out_size + w] = out_buf[i][h][w];
            }
        }
    }

}

}
