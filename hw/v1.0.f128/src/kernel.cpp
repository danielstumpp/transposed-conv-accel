#include "kernel.hpp"

#define MAX(A, B) ((A >= B) ? A : B)

/*
void TransposeConv2d_kernel(DTYPE in[CFG::in_channels][CFG::in_size][CFG::in_size],
                     DTYPE bias[CFG::out_channels],
                     DTYPE kernel[CFG::out_channels][CFG::in_channels][CFG::kernel_size][CFG::kernel_size],
                     DTYPE out[CFG::out_channels][CFG::out_size][CFG::out_size])
                     */

extern "C" {

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

    static const int inpad = MAX(CFG::kernel_size - CFG::pad - 1, 0);

    HWTYPE weights_block[CFG::ocTile][CFG::in_channels][CFG::kernel_size][CFG::kernel_size];
    HWTYPE out_block[CFG::ocTile][CFG::osTile][CFG::osTile];
    HWTYPE in_block[CFG::in_channels][CFG::osTile/CFG::stride][CFG::osTile/CFG::stride];

    #pragma HLS array_partition variable=weights_block dim=2 complete
    #pragma HLS array_partition variable=in_block dim=1 complete
    //#pragma HLS array_partition variable=out_block factor=2 cyclic
    //#pragma HLS array_partition variable=out_block dim=2 complete
    //#pragma HLS array_partition variable=out_block dim=3 complete

    for (int ht = 0; ht < CFG::out_size; ht += CFG::osTile){
        for (int wt = 0; wt < CFG::out_size; wt += CFG::osTile){
            for (int it = 0; it < CFG::out_channels; it += CFG::ocTile){

                // load bias into output block
                for (int i = 0; i < CFG::ocTile; ++i){
                    HWTYPE b = bias[it + i];
                    for (int h = 0; h < CFG::osTile; ++h){
                        for (int w = 0; w < CFG::osTile; ++w){
                            out_block[i][h][w] = b;
                        }
                    }
                }

                // load kernel weights
                for (int i = 0; i < CFG::ocTile; ++i){
                    for (int j = 0; j < CFG::in_channels; ++j){
                        for (int p = 0; p < CFG::kernel_size; ++p){
                            for (int q = 0; q < CFG::kernel_size; ++q){
                                weights_block[i][j][p][q] = kernel[((i + it) * CFG::in_channels * CFG::kernel_size * CFG::kernel_size) + (j * CFG::kernel_size * CFG::kernel_size) + (p * CFG::kernel_size) + q];
                            }
                        }
                    }
                }

                // load input features
                for (int h = 0; h < CFG::osTile; ++h){
                    for (int w = 0; w < CFG::osTile; ++w){
                        for (int j = 0; j < CFG::in_channels; ++j){
                            in_block[j][h / CFG::stride][w / CFG::stride] = in[j * CFG::in_size * CFG::in_size + ((h + ht)/ CFG::stride) * CFG::in_size + ((w + wt) / CFG::stride)];
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
                                            acc += weights_block[i][j][p][q] * in_block[j][h / CFG::stride][w / CFG::stride];
                                        }                           
                                    }
                                    out_block[i][h][w] += acc;
                                }
                            }
                        }
                    }
                }  


                // write output to DRAM
                for (int i = 0; i < CFG::ocTile; ++i){
                    for (int h = 0; h < CFG::osTile; ++h){
                        for (int w = 0; w < CFG::osTile; ++w){
                            out[(i + it) * CFG::out_size * CFG::out_size + (h + ht) * CFG::out_size + (w + wt)] = out_block[i][h][w];
                        }
                    }
                }
            }  
        }
    }                          

}

}
