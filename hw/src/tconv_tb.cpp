#include <stdio.h>
#include <iostream>
#include "TransposeConv2d.hpp"

template <typename T>
void init_mat(T *mat, const int size, const int m)
{
    for (int i = 0; i < size; ++i){
        mat[i] = (i % m) + 1;
    }
}

template <typename T>
void printmat(T *mat, const int size)
{
    for (int i = 0; i < size; ++i){
        for (int j = 0; j < size; ++j){
            std::cout << mat[i*size + j] <<"  ";
        }
        std::cout<<std::endl;
    }
    std::cout<<std::endl;
}


int main()
{

    static DTYPE in[CFG::in_size][CFG::in_size];
    static DTYPE kernel[CFG::kernel_size][CFG::kernel_size];
    static DTYPE out[CFG::out_size][CFG::out_size];

    init_mat(&in[0][0], CFG::in_size*CFG::in_size, 1);
    init_mat(&kernel[0][0], CFG::kernel_size*CFG::kernel_size, 1);

    std::cout << "in:\n"; printmat(&in[0][0], CFG::in_size);
    std::cout << "kernel:\n"; printmat(&kernel[0][0], CFG::kernel_size);

    TransposeConv2d(in, 0, kernel, out);

    std::cout << "out:\n"; printmat(&out[0][0], CFG::out_size);

    return EXIT_SUCCESS;
}