/**********
Copyright (c) 2019, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/

#include "xcl2.hpp"
#include <algorithm>
#include <vector>
#include <ctime>
#include <iostream>
#include "../config.hpp"
#include "kernel.hpp"
#include "TransposeConv2d.hpp"

void init_mat(std::vector<DTYPE, aligned_allocator<DTYPE>> &mat, const int size, const int m)
{
    for (int i = 0; i < size; ++i){
        mat[i] = (i % m) + 1;
    }
}

const bool check(std::vector<DTYPE, aligned_allocator<DTYPE>> &A, std::vector<DTYPE, aligned_allocator<DTYPE>> &B, const int M, const int N)
{
    for (int i = 0; i < M*N; i++) {
        if (A[i] != B[i])
            return false;
    }
    return true;
}

void printmat(std::vector<DTYPE, aligned_allocator<DTYPE>> &mat, const int size, const int channels, std::string name)
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

int main(int argc, char **argv) {
	if (argc != 2) {
		std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
		return EXIT_FAILURE;
	}

	std::string binaryFile = argv[1];
	cl_int err;
	cl::Context context;
	cl::Kernel krnl;
	cl::CommandQueue q;

	std::vector<DTYPE, aligned_allocator<DTYPE>> conv_in(CFG::in_channels*CFG::in_size*CFG::in_size);
	std::vector<DTYPE, aligned_allocator<DTYPE>> conv_bias(CFG::out_channels);
	std::vector<DTYPE, aligned_allocator<DTYPE>> conv_kernel(CFG::out_channels*CFG::in_channels*CFG::kernel_size*CFG::kernel_size);
	std::vector<DTYPE, aligned_allocator<DTYPE>> conv_out_hw(CFG::out_channels*CFG::out_size*CFG::out_size);
	std::vector<DTYPE, aligned_allocator<DTYPE>> conv_out_sw(CFG::out_channels*CFG::out_size*CFG::out_size);

	init_mat(conv_in, CFG::in_channels*CFG::in_size*CFG::in_size, 1);
    init_mat(conv_kernel,CFG::out_channels*CFG::in_channels*CFG::kernel_size*CFG::kernel_size, 1);
    init_mat(conv_bias, CFG::out_channels, 1);
    conv_bias[0] = 0;

    // Generate SW ground truth
    // TODO: Should we compare against 
   TransposeConv2d_arr(conv_in.data(), conv_bias.data(), conv_kernel.data(), conv_out_sw.data()); 


  // OPENCL HOST CODE AREA START
  // get_xil_devices() is a utility API which will find the xilinx
  // platforms and will return list of devices connected to Xilinx platform
  auto devices = xcl::get_xil_devices();
  // read_binary_file() is a utility API which will load the binaryFile
  // and will return the pointer to file buffer.
  auto fileBuf = xcl::read_binary_file(binaryFile);
  cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
  int valid_device = 0;
  for (unsigned int i = 0; i < devices.size(); i++) {
    auto device = devices[i];
    // Creating Context and Command Queue for selected Device
    OCL_CHECK(err, context = cl::Context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err, q = cl::CommandQueue(context, device,
                                        CL_QUEUE_PROFILING_ENABLE, &err));
    std::cout << "Trying to program device[" << i
              << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    cl::Program program(context, {device}, bins, NULL, &err);
    if (err != CL_SUCCESS) {
      std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
    } else {
      std::cout << "Device[" << i << "]: program successful!\n";
      OCL_CHECK(err, krnl = cl::Kernel(program, "TransposeConv2d_kernel", &err));
      valid_device++;
      break; // we break because we found a valid device
    }
  }
  if (valid_device == 0) {
    std::cout << "Failed to program any device found, exit!\n";
    exit(EXIT_FAILURE);
  }

  // Allocate Buffer in Global Memory
  // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and
  // Device-to-host communication
  OCL_CHECK(err, cl::Buffer buffer_in(
                     context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     CFG::in_channels*CFG::in_size*CFG::in_size,
                     conv_in.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_bias(
                     context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     CFG::out_channels,
                     conv_bias.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_conv_kernel(
                     context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     CFG::out_channels*CFG::in_channels*CFG::kernel_size*CFG::kernel_size,
                     conv_kernel.data(), &err));
  OCL_CHECK(err, cl::Buffer buffer_out(
                     context, CL_MEM_USE_HOST_PTR ,
                     CFG::out_channels*CFG::out_size*CFG::out_size,
                     conv_out_hw.data(), &err));

  OCL_CHECK(err, err = krnl.setArg(0, buffer_in));
  OCL_CHECK(err, err = krnl.setArg(1, buffer_bias));
  OCL_CHECK(err, err = krnl.setArg(2, buffer_conv_kernel));
  OCL_CHECK(err, err = krnl.setArg(3, buffer_out));

  // Copy input data to device global memory
  OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_in, buffer_bias, buffer_conv_kernel},
                                                  0 /* 0 means from host*/));

  q.finish();

  // Launch the Kernel
  // For HLS kernels global and local size is always (1,1,1). So, it is
  // recommended
  // to always use enqueueTask() for invoking HLS kernel

  const int n_trials = 1;
  double cumu_time = 0;
  auto trial_times = std::vector<double>();
  for (int i = 0; i < n_trials; i++) {
	  double kernel_time_in_sec = 0;
	  std::chrono::duration<double> kernel_time(0);
	  auto kernel_start = std::chrono::high_resolution_clock::now();

	  // Only want to time this kernel to determine bandwidth
	  OCL_CHECK(err, err = q.enqueueTask(krnl));
	  q.finish();

	  auto kernel_end = std::chrono::high_resolution_clock::now();
	  kernel_time = std::chrono::duration<double>(kernel_end - kernel_start);
	  kernel_time_in_sec = kernel_time.count();
	  cumu_time += kernel_time_in_sec;
	  trial_times.push_back(kernel_time_in_sec);

  // Copy Result from Device Global Memory to Host Local Memory
  OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_out},
                                                  CL_MIGRATE_MEM_OBJECT_HOST));
  q.finish();

  }

  auto avg_time = cumu_time / n_trials;
  auto min_time = *std::min_element(trial_times.begin(), trial_times.end()); 
  std::cout << "MIN Exection time after " << n_trials << " trials = " << min_time*1000 << "ms" << std::endl;
  std::cout << "AVG Exection time after " << n_trials << " trials = " << avg_time*1000 << "ms" << std::endl;


  // OPENCL HOST CODE AREA END
  std::cout << "Ground truth" << std::endl;
  printmat(conv_out_sw, CFG::out_size, CFG::out_channels, "conv out GT");

  std::cout << "Test"<<std::endl;
  printmat(conv_out_hw, CFG::out_size, CFG::out_channels, "conv out test");

  // Compare the results of the Device to the simulation
  bool match = check(conv_out_sw, conv_out_hw, 1, CFG::out_channels*CFG::out_size*CFG::out_size);

  std::cout << "TEST " << (match ? "PASSED" : "FAILED") << std::endl;
  return (match ? EXIT_SUCCESS : EXIT_FAILURE);
}
