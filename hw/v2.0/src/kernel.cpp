#include "kernel.hpp"
#include "ap_int.h"

#define MAX(A, B) ((A >= B) ? A : B)



extern "C" {

void TransposeConv2d_kernel(block_t *in, HWTYPE *bias, HWTYPE *kernel, HWTYPE *out)
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

    HWTYPE weights_block[CFG::ocTile][CFG::in_channels][CFG::kernel_size][CFG::kernel_size];
    HWTYPE out_block[CFG::ocTile][CFG::osTile][CFG::osTile];
    HWTYPE in_block[CFG::in_channels][CFG::osTile/CFG::stride][CFG::osTile/CFG::stride];

    #pragma HLS array_partition variable=weights_block dim=2 complete
    #pragma HLS array_partition variable=in_block dim=1 complete

    tile_h_loop: for (int ht = 0; ht < CFG::out_size; ht += CFG::osTile){
        tile_w_loop: for (int wt = 0; wt < CFG::out_size; wt += CFG::osTile){
            tile_i_loop: for (int it = 0; it < CFG::out_channels; it += CFG::ocTile){

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
                load_kernel_i_loop: for (int i = 0; i < CFG::ocTile; ++i){
                    load_kernel_j_loop: for (int j = 0; j < CFG::in_channels; ++j){
                        load_kernel_p_loop: for (int p = 0; p < CFG::kernel_size; ++p){
                            load_kernel_q_loop: for (int q = 0; q < CFG::kernel_size; ++q){
                                    weights_block[i][j][p][q] = kernel[((i + it) * CFG::in_channels * CFG::kernel_size * CFG::kernel_size) + (j * CFG::kernel_size * CFG::kernel_size) + (p * CFG::kernel_size) + q];
                            }
                        }
                    }
                }

                // load input features
                load_in_h_loop: for (int h = 0; h < CFG::osTile; ++h){
                    load_in_w_loop: for (int w = 0; w < CFG::osTile; ++w){
                        load_in_j_loop: for (int j = 0; j < CFG::in_channels; ++j){
                            load_in_p_loop: for (int p = 0; p < CFG::kernel_size; ++p) {
                                load_in_q_loop: for (int q = 0; q < CFG::kernel_size; ++q){
                                    load_in_zero_check: if ((h + ht + p) % CFG::stride == inpad && (w + wt + q) % CFG::stride == inpad){
                                        in_block[j][h / CFG::stride][w / CFG::stride] = in[j * CFG::in_size * CFG::in_size + ((h + ht) / CFG::stride) * CFG::in_size + ((w + wt) / CFG::stride)];
                                    }   
                                }
                            }
                        }
                    }
                }

                // perform convolution
                conv_p_loop: for (int p = 0; p < CFG::kernel_size; ++p){
                    conv_q_loop: for (int q = 0; q < CFG::kernel_size; ++q){
                        conv_h_loop: for (int h = 0; h < CFG::osTile; ++h){
                            conv_w_loop: for (int w = 0; w < CFG::osTile; ++w){
                                conv_i_loop: for (int i = 0; i < CFG::ocTile; ++i){
                                    #pragma HLS pipeline II=1
                                    HWTYPE acc = 0;
                                    conv_j_loop: for (int j = 0; j < CFG::in_channels; ++j){
                                        #pragma HLS unroll
                                        conv_zero_check: if ((h + ht + p) % CFG::stride == inpad && (w + wt + q) % CFG::stride == inpad){
                                            acc += weights_block[i][j][p][q] * in_block[j][h / CFG::stride][w / CFG::stride];
                                        }                           
                                    }
                                    out_block[i][h][w] = acc;
                                }
                            }
                        }
                    }
                } 

                // write output to DRAM
                write_out_i_loop: for (int i = 0; i < CFG::ocTile; ++i){
                    write_out_h_loop: for (int h = 0; h < CFG::osTile; ++h){                       
                        write_out_w_loop: for (int ww = 0; ww < CFG::osTile / OS_BW; ++ww){
                            #pragma HLS pipeline II=1
                            os_block_t out_temp;
                            for (int w = 0; w < OS_BW; ++w){
                                #pragma HLS unroll
                                out_temp(WORD_BITS * (w + 1) - 1, WORD_BITS * w) = out_block[i][h][ww * OS_BW + w];
                            }
                            out[(i + it) * CFG::out_size * CFG::out_size + (h + ht) * CFG::out_size / OS_BW + (ww + wt)] = out_temp;
                        }
                    }
                }
            }  
        }
    }            

}

}
