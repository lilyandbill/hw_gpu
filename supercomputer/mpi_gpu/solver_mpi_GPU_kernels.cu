#include "solver_mpi_GPU.h"

#include <cuda_runtime.h>

// ======================
// 5 点差分核：带 halo 的 u_halo -> 内部 Au
// ======================
__global__
void apply_A_kernel(int nx, int ny, int halo_nx,
                    const double* __restrict__ u_halo,
                    double* __restrict__ Au)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = nx * ny;
    if (idx >= N) return;

    int ii = idx % nx;
    int jj = idx / nx;

    // 对应 halo 中的 (ii+1, jj+1)
    int idh = (jj + 1) * halo_nx + (ii + 1);

    double wC = u_halo[idh];
    double wL = u_halo[idh - 1];
    double wR = u_halo[idh + 1];
    double wB = u_halo[idh - halo_nx];
    double wT = u_halo[idh + halo_nx];

    Au[idx] = (-4.0 * wC + wL + wR + wB + wT);
}

// 封装函数：给 C++ 用的接口（计算 grid/block 并 launch kernel）
void apply_A_cuda(int nx, int ny, int halo_nx,
                  const double* d_u_halo, double* d_Au)
{
    int N = nx * ny;
    int blockSize = 256;
    int gridSize  = (N + blockSize - 1) / blockSize;

    apply_A_kernel<<<gridSize, blockSize>>>(nx, ny, halo_nx,
                                            d_u_halo, d_Au);
    cudaDeviceSynchronize();
}

// GPU 缓冲区分配 / 释放（类的成员函数实现）
void MPISolver2D::allocate_gpu_buffers()
{
    if (gpu_buffers_allocated_) return;

    // 带 halo 的 u_halo，尺寸 halo_nx_ * halo_ny_
    size_t haloSize     = static_cast<size_t>(halo_nx_) * halo_ny_ * sizeof(double);
    // 内部 Au，尺寸 nx_ * ny_
    size_t interiorSize = static_cast<size_t>(nx_) * ny_ * sizeof(double);

    cudaMalloc(&d_u_halo_, haloSize);
    cudaMalloc(&d_Au_,     interiorSize);

    gpu_buffers_allocated_ = true;
}

void MPISolver2D::free_gpu_buffers()
{
    if (!gpu_buffers_allocated_) return;

    if (d_u_halo_) cudaFree(d_u_halo_);
    if (d_Au_)     cudaFree(d_Au_);

    d_u_halo_ = nullptr;
    d_Au_     = nullptr;
    gpu_buffers_allocated_ = false;
}

