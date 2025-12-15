# ================== 编译器与架构设置 ==================
# 主机端 MPI 编译器
HOST_COMP ?= mpicxx        # 或 mpicc，看你系统
CXX ?= mpicxx

# CUDA 编译器
NVCC      ?= nvcc

# GPU 架构
ARCH      ?= sm_90         

# 是否启用 GPU
USE_GPU   ?= 1

# ================== 编译选项 ==================
CXXFLAGS  ?= -std=c++11 -O2 -Wall
NVCCFLAGS ?= -std=c++11 -O2 -Xcompiler -Wall

#cuda路径
CUDA_PATH    ?= /usr/local/cuda
CUDA_INC_DIR ?= $(CUDA_PATH)/include
CUDA_LIB_DIR ?= $(CUDA_PATH)/lib64

INCLUDES  = -I. -I$(CUDA_INC_DIR)

ifeq ($(USE_GPU),1)
  CXXFLAGS  += -DUSE_CUDA
  NVCCFLAGS += -DUSE_CUDA -arch=$(ARCH) -ccbin $(HOST_COMP)
endif

# ================== 目标与源文件 ==================
TARGET = main_mpi_GPU      # 如果你叫 main_mpi_CUDA，就改成 main_mpi_CUDA

SRCS_CPP = main_mpi_GPU.cpp \
           domain_decomp_GPU.cpp \
           solver_mpi_GPU.cpp

SRCS_CU  = solver_mpi_GPU_kernels.cu

OBJS = $(SRCS_CPP:.cpp=.o) $(SRCS_CU:.cu=.o)

# ================== 默认目标 ==================
all: $(TARGET)

# 最终可执行文件：用 HOST_COMP 链接，并链接 CUDA 运行库
$(TARGET): $(OBJS)
	$(HOST_COMP) -o $@ $(OBJS) -L$(CUDA_LIB_DIR) -lcudart -lm -lstdc++
#$(HOST_COMP) -o $@ $(OBJS) -lcudart

# ================== 特殊规则 ==================
# 1) main_mpi_GPU.cpp 里用了 <cuda_runtime.h>，必须让 nvcc 来编译
#    （如果你的文件叫 main_mpi_CUDA.cpp，这里和上面的名字一起改）
main_mpi_GPU.o: main_mpi_GPU.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# ================== 通用规则 ==================
# 2) 其它普通 C++ 源文件：.cpp -> .o （不用 CUDA 头文件）
%.o: %.cpp
	$(HOST_COMP) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# 3) CUDA 源文件：.cu -> .o
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# ================== 辅助目标 ==================
clean:
	rm -f $(OBJS) $(TARGET)

run: $(TARGET)
	mpirun -np 4 ./$(TARGET) 40 40
