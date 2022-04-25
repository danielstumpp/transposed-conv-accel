#include <stdio.h>
#include <iostream>
#include <string>
#include "../src/TransposeConv2d.hpp"

template <typename T>
void init_mat(T *mat, const int size, const int m)
{
    for (int i = 0; i < size; ++i){
        mat[i] = (i % m) + 1;
    }
}

template<typename T>
const bool check(T* A, T* B, const int M, const int N)
{
    for (int i = 0; i < M*N; i++) {
        if (A[i] != B[i])
        {
            printf("%d != %d\n",A[i], B[i]);
            return false;
        }

            
    }
    return true;
}

template <typename T>
void printmat(T *mat, const int size, const int channels, std::string name)
{
    for (int c = 0; c < channels; ++c){
        std::cout<<name<<"["<<c<<"]:\n";
        for (int i = 0; i < size; ++i){
            for (int j = 0; j < size; ++j){
                std::cout << mat[c*size*size + i*size + j] <<"  ";
            }
            std::cout<<std::endl;
        }
        std::cout<<std::endl;
    }
}


int main()
{

    DTYPE *in = new DTYPE[CFG::in_channels*CFG::in_size*CFG::in_size];
    DTYPE *kernel = new DTYPE[CFG::out_channels*CFG::in_channels*CFG::kernel_size*CFG::kernel_size];
    DTYPE *out = new DTYPE[CFG::out_channels*CFG::out_size*CFG::out_size];
    DTYPE *bias = new DTYPE[CFG::out_channels];

    auto out_test = new DTYPE[CFG::out_channels*CFG::out_size*CFG::out_size];

    init_mat(in, CFG::in_channels*CFG::in_size*CFG::in_size, 3);
    init_mat(kernel,CFG::out_channels*CFG::in_channels*CFG::kernel_size*CFG::kernel_size, 6);
    init_mat(bias, CFG::out_channels, 5);


    // printmat(in_ptr, CFG::in_size, CFG::in_channels, "in");

    TransposeConv2d_arr_gold(in, bias, kernel, out);
    TransposeConv2d_arr(in, bias, kernel, out_test);

    // printmat(out_ptr, CFG::out_size, CFG::out_channels, "out");
    // printmat(out_test, CFG::out_size, CFG::out_channels, "out_test");

    std::cout << "kernel " << (check(out, out_test, 1, CFG::out_channels*CFG::out_size*CFG::out_size) ? "PASSED" : "FAILED" ) << std::endl;

    delete[] out_test;
    delete[] in;
    delete[] kernel;
    delete[] out;
    delete[] bias;

    return EXIT_SUCCESS;
}
