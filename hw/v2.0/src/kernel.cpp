#include "kernel.hpp"
#include <stdio.h>

#define MAX(A, B) ((A >= B) ? A : B)

/*
void TransposeConv2d_kernel(DTYPE in[CFG::in_channels][CFG::in_size][CFG::in_size],
                     DTYPE bias[CFG::out_channels],
                     DTYPE kernel[CFG::out_channels][CFG::in_channels][CFG::kernel_size][CFG::kernel_size],
                     DTYPE out[CFG::out_channels][CFG::out_size][CFG::out_size])
                     */

extern "C" {

void TransposeConv2d_kernel(block512_t *in, block256_t *bias, block512_t *kernel, block256_t *out)
{
#pragma HLS interface m_axi port = in bundle = gmem0
#pragma HLS interface m_axi port = bias bundle = gmem1
#pragma HLS interface m_axi port = kernel bundle = gmem2
#pragma HLS interface m_axi port = out bundle = gmem1
#pragma HLS interface s_axilite port = in bundle = control
#pragma HLS interface s_axilite port = bias bundle = control
#pragma HLS interface s_axilite port = kernel bundle = control
#pragma HLS interface s_axilite port = out bundle = control
#pragma HLS interface s_axilite port = return bundle = control

    static const int inpad = MAX(CFG::kernel_size - CFG::pad - 1, 0);

    HWTYPE weights_block[CFG::ocTile][CFG::kernel_size][CFG::kernel_size][CFG::in_channels];
    HWTYPE out_block[CFG::osTile][CFG::osTile][CFG::ocTile];
    HWTYPE in_block[CFG::osTile / CFG::stride][CFG::osTile / CFG::stride][CFG::in_channels];

    #pragma HLS array_partition variable=weights_block dim=4 complete
    #pragma HLS array_partition variable=in_block dim=3 complete
    #pragma HLS array_partition variable=out_block dim=3 complete

    for (int ht = 0; ht < CFG::out_size; ht += CFG::osTile){
        for (int wt = 0; wt < CFG::out_size; wt += CFG::osTile){
            for (int it = 0; it < CFG::out_channels; it += CFG::ocTile){

                // load bias into output block
                for (int h = 0; h < CFG::osTile; ++h){
                    for (int w = 0; w < CFG::osTile; ++w){
                        for (int ii = 0; ii < CFG::ocTile/WIDTH256; ++ii){
                            #pragma HLS pipeline II=1
                            block256_t b_temp = bias[it/WIDTH256 + ii];
                            for (int i = 0; i < WIDTH256; ++i){
                                #pragma HLS unroll
                                HWTYPE b = (HWTYPE)b_temp(WORD_BITS * (i + 1) - 1, WORD_BITS * i);
                                out_block[h][w][ii*WIDTH256 + i] = b;
                            }
                        }
                    } 
                }

                // load kernel weights
                for (int i = 0; i < CFG::ocTile; ++i){
                    for (int p = 0; p < CFG::kernel_size; ++p){
                        for (int q = 0; q < CFG::kernel_size; ++q){
                            for (int jj = 0; jj < CFG::in_channels/WIDTH512; ++jj){
                                #pragma HLS pipeline II=1
                                block512_t weights_temp = kernel[((i + it) * CFG::in_channels * CFG::kernel_size * CFG::kernel_size / WIDTH512) + (p * CFG::kernel_size * CFG::in_channels / WIDTH512) + (p * CFG::in_channels / WIDTH512) + jj];
                                for (int j = 0; j < WIDTH512; ++j){
                                    #pragma HLS unroll
                                    ap_int<16> val = weights_temp(WORD_BITS * (j + 1) - 1, WORD_BITS * j);
                                    weights_block[i][p][q][jj*WIDTH512 + j] = (HWTYPE) val;
                                }
                            }
                        }
                    }
                }

                // load input features
                for (int h = 0; h < CFG::osTile; ++h){
                    for (int w = 0; w < CFG::osTile; ++w){
                        for (int jj = 0; jj < CFG::in_channels/WIDTH512; ++jj){
                            #pragma HLS pipeline II=1
                            block512_t in_temp = in[((h + ht) / CFG::stride) * CFG::in_size * CFG::in_channels/WIDTH512 + ((w + wt) / CFG::stride) * CFG::in_channels/WIDTH512 + jj];
                            for (int j = 0; j < WIDTH512; ++j){
                                #pragma HLS unroll
                                ap_int<16> val = in_temp(WORD_BITS * (j + 1) - 1, WORD_BITS * j);
                                in_block[h / CFG::stride][w / CFG::stride][jj*WIDTH512 + j] = (HWTYPE) val;
                            }
                        }
                    }
                }
         

                // perform convolution
                for (int p = 0; p < CFG::kernel_size; ++p) {
                    for (int q = 0; q < CFG::kernel_size; ++q){
                        for (int h = 0; h < CFG::osTile; ++h){
                            for (int w = 0; w < CFG::osTile; ++w){
                                for (int i = 0; i < CFG::ocTile; ++i){
                                    #pragma HLS pipeline II=1
                                    HWTYPE acc = 0;
                                    for (int j = 0; j < CFG::in_channels; ++j){
                                        #pragma HLS unroll
                                        if ((h + ht + p) % CFG::stride == inpad && (w + wt + q) % CFG::stride == inpad){
                                            acc += weights_block[i][p][q][j] * in_block[h / CFG::stride][w / CFG::stride][j];
                                        }                           
                                    }
                                    out_block[h][w][i] += acc;
                                }
                            }
                        }
                    }
                }  


                // write output to DRAM
                for (int h = 0; h < CFG::osTile; ++h){
                    for (int w = 0; w < CFG::osTile; ++w){
                        for (int ii = 0; ii < CFG::ocTile/WIDTH256; ++ii){
                            #pragma HLS pipeline II=1
                            block256_t out_temp;
                            for (int i = 0; i < WIDTH256; ++i){
                                #pragma HLS unroll
                                out_temp(WORD_BITS * (i+1) - 1, WORD_BITS*i) = out_block[h][w][ii*WIDTH256 + i];
                            }
                            out[(h + ht) * CFG::out_size * CFG::out_channels/WIDTH256 + (w + wt) * CFG::out_channels/WIDTH256 + (ii + it/WIDTH256)] = out_temp;
                        }
                    }
                }
            }  
        }
    }                          

}
}
