#!/bin/sh
lli=${LLVMINTERP-lli}
exec $lli \
    /ihome/ageorge/cjg91/dev/transposed-conv-accel/hw/src/_x.sw_emu.xilinx_u200_xdma_201830_2.kernel.cpp/transpose_conv/TransposeConv2d_kernel/TransposeConv2d_kernel/solution/.autopilot/db/a.g.bc ${1+"$@"}
