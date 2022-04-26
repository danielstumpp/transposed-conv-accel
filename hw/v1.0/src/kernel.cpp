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

    tile_h_loop: for (int ht = 0; ht < CFG::out_size; ht += CFG::osTile){
        tile_w_loop: for (int wt = 0; wt < CFG::out_size; wt += CFG::osTile){
            tile_i_loop: for (int it = 0; it < CFG::out_channels; it += CFG::ocTile){

                // load bias into output block
                load_bias_h_loop: for (int h = ht; h < MIN(ht + CFG::osTile, CFG::out_size); ++h){
                    load_bias_w_loop: for (int w = wt; w < MIN(wt + CFG::osTile, CFG::out_size); ++w){
                        load_bias_i_loop: for (int i = it; i < MIN(it + CFG::ocTile, CFG::out_channels); ++i){
                            out_block[i - it][h - ht][w - wt] = bias[i];
                        }
                    }
                }

                // load kernel weights
                load_kernel_i_loop: for (int i = it; i < MIN(it + CFG::ocTile, CFG::out_channels); ++i){
                    load_kernel_j_loop: for (int j = 0; j < CFG::in_channels; ++j){
                        load_kernel_p_loop: for (int p = 0; p < CFG::kernel_size; ++p){
                            load_kernel_q_loop: for (int q = 0; q < CFG::kernel_size; ++q){
                                weights_block[i - it][j][p][q] = kernel[(i * CFG::in_channels * CFG::kernel_size * CFG::kernel_size) + (j * CFG::kernel_size * CFG::kernel_size) + (p * CFG::kernel_size) + q];
                            }
                        }
                    }
                }

                // load input features
                load_in_h_loop: for (int h = ht; h < MIN(ht + CFG::osTile, CFG::out_size); ++h){
                    load_in_w_loop: for (int w = wt; w < MIN(wt + CFG::osTile, CFG::out_size); ++w){
                        load_in_j_loop: for (int j = 0; j < CFG::in_channels; ++j){
                            load_in_p_loop: for (int p = 0; p < CFG::kernel_size; ++p) {
                                load_in_q_loop: for (int q = 0; q < CFG::kernel_size; ++q){
                                    load_in_zero_check: if ((h + p) % CFG::stride == inpad && (w+q) % CFG::stride == inpad){
                                        in_block[j][(h - ht) / CFG::stride][(w - wt) / CFG::stride] = in[j * CFG::in_size * CFG::in_size + (h / CFG::stride) * CFG::in_size + (w / CFG::stride)];
                                    }   
                                }
                            }
                        }
                    }
                }

                // perform convolution
                conv_p_loop: for (int p = 0; p < CFG::kernel_size; ++p){
                    conv_q_loop: for (int q = 0; q < CFG::kernel_size; ++q){
                        conv_h_loop: for (int h = ht; h < MIN(ht + CFG::osTile, CFG::out_size); ++h){
                            conv_w_loop: for (int w = wt; w < MIN(wt + CFG::osTile, CFG::out_size); ++w){
                                conv_i_loop: for (int i = it; i < MIN(it + CFG::ocTile, CFG::out_channels); ++i){
                                    #pragma HLS pipeline II=1
                                    conv_j_loop: for (int j = 0; j < CFG::in_channels; ++j){
                                        #pragma HLS unroll
                                        conv_zero_check: if ((h + p) % CFG::stride == inpad && (w+q) % CFG::stride == inpad){
                                            out_block[i - it][h - ht][w - wt] += weights_block[i - it][j][p][q] * in_block[j][(h - ht) / CFG::stride][(w - wt) / CFG::stride];
                                        }                           
                                    }
                                }
                            }
                        }
                    }
                } 

                // write output to DRAM
                write_h_loop: for (int h = ht; h < MIN(ht + CFG::osTile, CFG::out_size); ++h){
                    write_w_loop: for (int w = wt; w < MIN(wt + CFG::osTile, CFG::out_size); ++w){
                        write_i_loop: for (int i = it; i < MIN(it + CFG::ocTile, CFG::out_channels); ++i){
                            out[i * CFG::out_size * CFG::out_size + h * CFG::out_size + w] = out_block[i - it][h - ht][w - wt];
                        }
                    }
                }
            }  
        }
    }            

}

}
