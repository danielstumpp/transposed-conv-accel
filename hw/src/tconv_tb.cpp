#include <stdio.h>
#include <iostream>
#include <string>
#include "TransposeConv2d.hpp"

template <typename T>
void init_mat(T *mat, const int size, const int m)
{
    for (int i = 0; i < size; ++i){
        mat[i] = (i % m) + 1;
    }
}

template <typename T>
void printmat(T *mat, const int size, const int channels, std::string name)
{
    for (int c = 0; c < channels; ++c){
        std::cout<<name<<"["<<c<<"]:\n";
        for (int i = 0; i < size; ++i){
            for (int j = 0; j < size; ++j){
                std::cout << mat[i*size + j] <<"  ";
            }
            std::cout<<std::endl;
        }
        std::cout<<std::endl;
    }
    
}


int main()
{

    static DTYPE in[CFG::in_channels][CFG::in_size][CFG::in_size];
    static DTYPE kernel[CFG::out_channels][CFG::in_channels][CFG::kernel_size][CFG::kernel_size];
    static DTYPE out[CFG::out_channels][CFG::out_size][CFG::out_size];
    static DTYPE bias[CFG::out_channels];

    DTYPE *in_ptr = &in[0][0][0];
    DTYPE *kernel_ptr = &kernel[0][0][0][0];
    DTYPE *out_ptr = &out[0][0][0];
    DTYPE *bias_ptr = &bias[0];

    init_mat(in_ptr, CFG::in_channels*CFG::in_size*CFG::in_size, 1);
    init_mat(kernel_ptr,CFG::out_channels*CFG::in_channels*CFG::kernel_size*CFG::kernel_size, 1);
    init_mat(bias_ptr, CFG::out_channels, 1);

    printmat(in_ptr, CFG::in_size, CFG::in_channels, "in");

    TransposeConv2d(in, bias, kernel, out);

    printmat(out_ptr, CFG::out_size, CFG::out_channels, "out");

    return EXIT_SUCCESS;
}