#include "TransposeConv2d.hpp"
#include <iostream>

void TransposeConv2d(DTYPE in[CFG::in_channels][CFG::in_size][CFG::in_size],
                     DTYPE bias[CFG::out_channels],
                     DTYPE kernel[CFG::out_channels][CFG::in_channels][CFG::kernel_size][CFG::kernel_size],
                     DTYPE out[CFG::out_channels][CFG::out_size][CFG::out_size])
{

    constexpr int pin_size = CFG::out_size + CFG::kernel_size - 1;
    DTYPE padded_in[CFG::in_channels][pin_size][pin_size];
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
                padded_in[c][CFG::pad + h*CFG::stride][CFG::pad + w*CFG::stride] = in[c][h][w]; 
            }
        }
    }

    for (int i = 0; i < CFG::out_channels; ++i){
        for (int h = 0; h < CFG::out_size; ++h){
            for (int w = 0; w < CFG::out_size; ++w){
                out[i][h][w] = bias[i];
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
                            out[i][h][w] += kernel[i][j][p][q] * padded_in[j][h + p][w + q];
                        }
                    }
                }
            }
        }
    }
    
}