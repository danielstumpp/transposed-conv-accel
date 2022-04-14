#!/bin/sh

# 
# v++(TM)
# runme.sh: a v++-generated Runs Script for UNIX
# Copyright 1986-2019 Xilinx, Inc. All Rights Reserved.
# 

if [ -z "$PATH" ]; then
  PATH=/ihome/crc/install/xilinx/Vivado/2019.2/bin:/ihome/crc/install/xilinx/Vitis/2019.2/bin:/ihome/crc/install/xilinx/Vitis/2019.2/bin
else
  PATH=/ihome/crc/install/xilinx/Vivado/2019.2/bin:/ihome/crc/install/xilinx/Vitis/2019.2/bin:/ihome/crc/install/xilinx/Vitis/2019.2/bin:$PATH
fi
export PATH

if [ -z "$LD_LIBRARY_PATH" ]; then
  LD_LIBRARY_PATH=
else
  LD_LIBRARY_PATH=:$LD_LIBRARY_PATH
fi
export LD_LIBRARY_PATH

HD_PWD='/ihome/ageorge/cjg91/dev/transposed-conv-accel/hw/src/_x.sw_emu.xilinx_u200_xdma_201830_2.kernel.cpp/transpose_conv/TransposeConv2d_kernel'
cd "$HD_PWD"

HD_LOG=runme.log
/bin/touch $HD_LOG

ISEStep="./ISEWrap.sh"
EAStep()
{
     $ISEStep $HD_LOG "$@" >> $HD_LOG 2>&1
     if [ $? -ne 0 ]
     then
         exit
     fi
}

EAStep vivado_hls -f TransposeConv2d_kernel.tcl -messageDb vivado_hls.pb
