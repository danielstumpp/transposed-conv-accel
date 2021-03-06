DEVICE := xilinx_u200_xdma_201830_2

# Points to top directory of Git repository
# TODO: make more general
COMMON_REPO = ../
PWD = $(shell readlink -f .)
ABS_COMMON_REPO = $(shell readlink -f $(COMMON_REPO))

TARGET := hw
HOST_ARCH := x86
SYSROOT := 
CLK := 300000000

include ./utils.mk

XSA := $(call device2xsa, $(DEVICE))

KERNEL_IMPL := ./src/kernel.cpp

TEMP_DIR := ./_x.$(TARGET)
BUILD_DIR := ./build_dir.$(TARGET)

VPP := v++
SDCARD := sd_card

#Include Libraries
include $(ABS_COMMON_REPO)/common/includes/opencl/opencl.mk
include $(ABS_COMMON_REPO)/common/includes/xcl2/xcl2.mk
CXXFLAGS += $(xcl2_CXXFLAGS)
LDFLAGS += $(xcl2_LDFLAGS)
HOST_SRCS += $(xcl2_SRCS)
CXXFLAGS += $(opencl_CXXFLAGS) -Wall -O0 -g -std=c++11
LDFLAGS += $(opencl_LDFLAGS)

HOST_SRCS += ./src/host.cpp ../src/TransposeConv2d.cpp

# Host compiler global settings
CXXFLAGS += -fmessage-length=0 -std=c++0x
LDFLAGS += -lrt -lstdc++ 

ifneq ($(HOST_ARCH), x86)
	LDFLAGS += --sysroot=$(SYSROOT)
endif

# Kernel compiler global settings
CLFLAGS += -t $(TARGET) --platform $(DEVICE) --save-temps 
ifneq ($(TARGET), hw)
	CLFLAGS += -g
endif

ifneq ($(CFG),)
	LDCLFLAGS += --config $(CFG)
endif



EXECUTABLE = host
CMD_ARGS = $(BUILD_DIR)/tranpose_conv.xclbin
EMCONFIG_DIR = $(TEMP_DIR)
EMU_DIR = $(SDCARD)/data/emulation

BINARY_CONTAINERS += $(BUILD_DIR)/transpose_conv.xclbin
BINARY_CONTAINER_kernel_OBJS += $(TEMP_DIR)/transpose_conv.xo
RES_DIR := artifacts/

CP = cp -rf

.PHONY: all clean cleanall docs emconfig
all: check-devices $(EXECUTABLE) $(BINARY_CONTAINERS) emconfig sd_card 
ifneq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
	mkdir -p $(RES_DIR)
	cp $(EXECUTABLE) $(RES_DIR)
	cp $(BUILD_DIR)/transpose_conv.xclbin $(RES_DIR)
	cp $(BUILD_DIR)/reports/link/imp/kernel_util_routed.rpt $(RES_DIR) 
	cp $(BUILD_DIR)/reports/link/imp/xilinx_u200_xdma_201830_2_bb_locked_timing_summary_routed.rpt $(RES_DIR) 
	cp $(TEMP_DIR)/reports/transpose_conv/hls_reports/TransposeConv2d_kernel_csynth.rpt $(RES_DIR)
	cp $(TEMP_DIR)/logs/transpose_conv/TransposeConv2d_kernel_vivado_hls.log $(RES_DIR)	
endif
.PHONY: exe
exe: $(EXECUTABLE)

.PHONY: build
build: $(BINARY_CONTAINERS)


# Building kernel
$(TEMP_DIR)/transpose_conv.xo: $(KERNEL_IMPL)
	mkdir -p $(TEMP_DIR)
	$(VPP) $(CLFLAGS) --temp_dir $(TEMP_DIR) -c -k TransposeConv2d_kernel --hls.clock $(CLK)Hz:TransposeConv2d_kernel -I'$(<D)' -o'$@' '$<' 
$(BUILD_DIR)/transpose_conv.xclbin: $(BINARY_CONTAINER_kernel_OBJS)
	mkdir -p $(BUILD_DIR)
	$(VPP) $(CLFLAGS) --temp_dir $(BUILD_DIR) -l $(LDCLFLAGS) -o'$@' $(+)

# Building Host
$(EXECUTABLE): check-xrt $(HOST_SRCS) $(HOST_HDRS)
	$(CXX) $(CXXFLAGS) $(HOST_SRCS) $(HOST_HDRS) -o '$@' $(LDFLAGS)

emconfig:$(EMCONFIG_DIR)/emconfig.json
$(EMCONFIG_DIR)/emconfig.json:
	emconfigutil --platform $(DEVICE) --od $(EMCONFIG_DIR)

check: all
ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
ifeq ($(HOST_ARCH), x86)
	$(CP) $(EMCONFIG_DIR)/emconfig.json .
	XCL_EMULATION_MODE=$(TARGET) ./$(EXECUTABLE) $(BUILD_DIR)/transpose_conv.xclbin
else
	mkdir -p $(EMU_DIR)
	$(CP) $(XILINX_VITIS)/data/emulation/unified $(EMU_DIR)
	mkfatimg $(SDCARD) $(SDCARD).img 500000
	launch_emulator -no-reboot -runtime ocl -t $(TARGET) -sd-card-image $(SDCARD).img -device-family $(DEV_FAM)
endif
else
ifeq ($(HOST_ARCH), x86)
	./$(EXECUTABLE) $(BUILD_DIR)/transpose_conv.xclbin
endif
endif

.PHONY: test
test: $(EXECUTABLE)
ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
ifeq ($(HOST_ARCH), x86)
	XCL_EMULATION_MODE=$(TARGET) ./$(EXECUTABLE) $(BUILD_DIR)/transpose_conv.xclbin
else
	mkdir -p $(EMU_DIR)
	$(CP) $(XILINX_VITIS)/data/emulation/unified $(EMU_DIR)
	mkfatimg $(SDCARD) $(SDCARD).img 500000
	launch_emulator -no-reboot -runtime ocl -t $(TARGET) -sd-card-image $(SDCARD).img -device-family $(DEV_FAM)
endif
else
ifeq ($(HOST_ARCH), x86)
	./$(EXECUTABLE) $(BUILD_DIR)/transpose_conv.xclbin
else
	$(ECHO) "Please copy the content of sd_card folder and data to an SD Card and run on the board"
endif
endif


sd_card: $(EXECUTABLE) $(BINARY_CONTAINERS) emconfig
ifneq ($(HOST_ARCH), x86)
	mkdir -p $(SDCARD)/$(BUILD_DIR)
	$(CP) $(B_NAME)/sw/$(XSA)/boot/generic.readme $(B_NAME)/sw/$(XSA)/xrt/image/* xrt.ini $(EXECUTABLE) $(SDCARD)
	$(CP) $(BUILD_DIR)/*.xclbin $(SDCARD)/$(BUILD_DIR)/
ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
	$(ECHO) 'cd /mnt/' >> $(SDCARD)/init.sh
	$(ECHO) 'export XILINX_VITIS=$$PWD' >> $(SDCARD)/init.sh
	$(ECHO) 'export XCL_EMULATION_MODE=$(TARGET)' >> $(SDCARD)/init.sh
	$(ECHO) './$(EXECUTABLE) $(CMD_ARGS)' >> $(SDCARD)/init.sh
	$(ECHO) 'reboot' >> $(SDCARD)/init.sh
else
	[ -f $(SDCARD)/BOOT.BIN ] && echo "INFO: BOOT.BIN already exists" || $(CP) $(BUILD_DIR)/sd_card/BOOT.BIN $(SDCARD)/
	$(ECHO) 'cd /mnt/' >> $(SDCARD)/init.sh
	$(ECHO) './$(EXECUTABLE) $(CMD_ARGS)' >> $(SDCARD)/init.sh
endif
endif

# Cleaning stuff
clean:
	-$(RMDIR) $(EXECUTABLE) $(XCLBIN)/{*sw_emu*,*hw_emu*} 
	-$(RMDIR) profile_* TempConfig system_estimate.xtxt *.rpt *.csv 
	-$(RMDIR) src/*.ll *v++* .Xil emconfig.json dltmp* xmltmp* *.log *.jou *.wcfg *.wdb

cleanall: clean
	-$(RMDIR) build_dir* sd_card*
	-$(RMDIR) _x.* *xclbin.run_summary qemu-memory-_* emulation/ _vimage/ pl* start_simulation.sh *.xclbin

