#include "TransposeConv2d.hpp"

void TransposeConv2d(DTYPE in[CFG::in_size][CFG::in_size],
                     DTYPE bias,
                     DTYPE kernel[CFG::kernel_size][CFG::kernel_size],
                     DTYPE out[CFG::out_size][CFG::out_size])
{

    constexpr int pin_size = CFG::out_size + CFG::kernel_size - 1;
    DTYPE padded_in[pin_size][pin_size];
    for (int i = 0; i < pin_size; ++i){
        for (int j = 0; j < pin_size; ++j){
            padded_in[i][j] = 0;
        }
    }

    for (int h = 0; h < CFG::in_size; ++h){
        for (int w = 0; w < CFG::in_size; ++w){
            padded_in[CFG::pad + h*CFG::stride][CFG::pad + w*CFG::stride] = in[h][w]; 
        }
    }

    for (int h = 0; h < CFG::out_size; ++h){
        for (int w = 0; w < CFG::out_size; ++w) {
            for (int p = 0; p < CFG::kernel_size; ++p) {
                for (int q = 0; q < CFG::kernel_size; ++q)
                {
                    out[h][w] += kernel[p][q] * padded_in[h + p][w + q];
                }
            }
        }
    }
}