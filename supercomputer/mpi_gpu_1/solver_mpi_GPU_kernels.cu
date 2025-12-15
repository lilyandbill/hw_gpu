// // ================================================================
// solver_mpi_GPU_kernels.cu – FINAL VERSION
// ================================================================

#include "solver_mpi_GPU.h"
#include <cuda_runtime.h>
#include <algorithm>

// ================================================================
// Part 5 – 对外接口声明（必须位于所有 kernel 之前）
// ================================================================

// -------- vector operations --------
void vec_copy_cuda(double* d_y, const double* d_x,
                   int n, cudaStream_t stream);
void axpy_cuda(double* d_y, const double* d_x,
               double a, int n, cudaStream_t stream);
void scale_cuda(double* d_x,
                double a, int n, cudaStream_t stream);
void diag_solve_cuda(double* d_z,
                     const double* d_r,
                     const double* d_D,
                     int n, cudaStream_t stream);

// -------- dot product --------------
double dot_cuda_device(const double* d_a,
                       const double* d_b,
                       int n,
                       cudaStream_t stream);

// -------- halo exchange kernels -----
void pack_halo_cuda(
    const double* d_u,
    int nx, int ny,
    double* d_left, double* d_right,
    double* d_bottom, double* d_top,
    cudaStream_t stream);

void unpack_halo_cuda(
    double* d_u_halo,
    const double* d_left, const double* d_right,
    const double* d_bottom, const double* d_top,
    int nx, int ny,
    int halo_nx, int halo_ny,
    cudaStream_t stream);

void build_halo_interior_cuda(
    double* d_u_halo,
    const double* d_u,
    int nx, int ny,
    int halo_nx, int halo_ny,
    cudaStream_t stream);

// -------- apply_A (5-point stencil) --------
void apply_A_cuda_device(
    int nx, int ny, int halo_nx,
    const double* d_u_halo,
    double* d_Ap,
    cudaStream_t stream);


// ============================================================================
// Part 1 – 基础向量 Kernels：copy / axpy / scale / diag_solve
// ============================================================================

// ----------------------------------------------------------
// vec_copy: y = x
// ----------------------------------------------------------
__global__
void vec_copy_kernel(double* __restrict__ y,
                     const double* __restrict__ x,
                     int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        y[idx] = x[idx];
}

void vec_copy_cuda(double* d_y, const double* d_x,
                   int n, cudaStream_t stream)
{
    int block = 256;
    int grid  = (n + block - 1) / block;
    vec_copy_kernel<<<grid, block, 0, stream>>>(d_y, d_x, n);
}


// ----------------------------------------------------------
// axpy: y = y + a*x
// ----------------------------------------------------------
__global__
void axpy_kernel(double* __restrict__ y,
                 const double* __restrict__ x,
                 double a,
                 int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        y[idx] += a * x[idx];
}

void axpy_cuda(double* d_y, const double* d_x,
               double a, int n, cudaStream_t stream)
{
    int block = 256;
    int grid  = (n + block - 1) / block;
    axpy_kernel<<<grid, block, 0, stream>>>(d_y, d_x, a, n);
}


// ----------------------------------------------------------
// scale: x = a*x （用于 p = beta*p）
// ----------------------------------------------------------
__global__
void scale_kernel(double* __restrict__ x,
                  double a,
                  int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        x[idx] *= a;
}

void scale_cuda(double* d_x, double a, int n, cudaStream_t stream)
{
    int block = 256;
    int grid  = (n + block - 1) / block;
    scale_kernel<<<grid, block, 0, stream>>>(d_x, a, n);
}


// ----------------------------------------------------------
// diag_solve: z = r / D
// ----------------------------------------------------------
__global__
void diag_solve_kernel(double* __restrict__ z,
                       const double* __restrict__ r,
                       const double* __restrict__ D,
                       int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        z[idx] = r[idx] / D[idx];
}

void diag_solve_cuda(double* d_z,
                     const double* d_r,
                     const double* d_D,
                     int n,
                     cudaStream_t stream)
{
    int block = 256;
    int grid  = (n + block - 1) / block;
    diag_solve_kernel<<<grid, block, 0, stream>>>(d_z, d_r, d_D, n);
}


// ============================================================================
// Part 2 – dot reduce Kernels（elemwise + global pairwise reduction）
// ============================================================================

// ---- 元素乘法：tmp[i] = a[i] * b[i] ----
__global__
void elemwise_mul_kernel(const double* __restrict__ a,
                         const double* __restrict__ b,
                         double* __restrict__ out,
                         int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        out[idx] = a[idx] * b[idx];
}

// ---- pairwise 规约：data[i] += data[i+stride] ----
__global__
void reduce_pairwise_kernel(double* __restrict__ data,
                            int n,
                            int stride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int j = idx + stride;

    if (idx < stride && j < n)
        data[idx] += data[j];
}

// ---- GPU dot (device-only) ----
double dot_cuda_device(const double* d_a,
                       const double* d_b,
                       int n,
                       cudaStream_t stream)
{
    if (n <= 0) return 0.0;

    static double* d_buf = nullptr;
    static int buf_size = 0;

    // 分配或扩容 d_buf
    if (n > buf_size) {
        if (d_buf) cudaFree(d_buf);
        cudaMalloc(&d_buf, n * sizeof(double));
        buf_size = n;
    }

    // 1) 元素乘法
    {
        int block = 256;
        int grid  = (n + block - 1) / block;
        elemwise_mul_kernel<<<grid, block, 0, stream>>>(d_a, d_b, d_buf, n);
    }

    // 2) pairwise 规约
    int len = n;
    while (len > 1) {
        int stride = (len + 1) / 2;
        int block  = 256;
        int grid   = (stride + block - 1) / block;

        reduce_pairwise_kernel<<<grid, block, 0, stream>>>(d_buf, len, stride);

        len = stride;
    }

    // 3) 复制回 host
    double result = 0.0;
    cudaMemcpyAsync(&result, d_buf,
                    sizeof(double),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    return result;
}


// ============================================================================
// Part 3 – Halo Kernels（pack / unpack / interior build）
// ============================================================================

// ---- pack halo ----
__global__
void pack_halo_kernel(
    const double* __restrict__ d_u,
    int nx, int ny,
    double* __restrict__ left,
    double* __restrict__ right,
    double* __restrict__ bottom,
    double* __restrict__ top)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < ny) {
        left[idx]  = d_u[idx * nx + 0];
        right[idx] = d_u[idx * nx + (nx - 1)];
    }

    if (idx < nx) {
        bottom[idx] = d_u[idx];
        top[idx]    = d_u[(ny - 1) * nx + idx];
    }
}

void pack_halo_cuda(
    const double* d_u,
    int nx, int ny,
    double* d_left, double* d_right,
    double* d_bottom, double* d_top,
    cudaStream_t stream)
{
    int block = 256;
    int grid  = (max(nx, ny) + block - 1) / block;

    pack_halo_kernel<<<grid, block, 0, stream>>>(
        d_u, nx, ny,
        d_left, d_right,
        d_bottom, d_top);
}


// ---- unpack halo ----
__global__
void unpack_halo_kernel(
    double* __restrict__ d_u_halo,
    const double* __restrict__ left,
    const double* __restrict__ right,
    const double* __restrict__ bottom,
    const double* __restrict__ top,
    int nx, int ny,
    int halo_nx, int halo_ny)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < ny) {
        int row = idx + 1;
        d_u_halo[row * halo_nx + 0]       = left[idx];
        d_u_halo[row * halo_nx + (nx+1)]  = right[idx];
    }

    if (idx < nx) {
        int col = idx + 1;
        d_u_halo[0 * halo_nx + col]         = bottom[idx];
        d_u_halo[(ny+1) * halo_nx + col]    = top[idx];
    }
}

void unpack_halo_cuda(
    double* d_u_halo,
    const double* d_left, const double* d_right,
    const double* d_bottom, const double* d_top,
    int nx, int ny,
    int halo_nx, int halo_ny,
    cudaStream_t stream)
{
    int block = 256;
    int grid  = (max(nx, ny) + block - 1) / block;

    unpack_halo_kernel<<<grid, block, 0, stream>>>(
        d_u_halo,
        d_left, d_right,
        d_bottom, d_top,
        nx, ny,
        halo_nx, halo_ny);
}


// ---- build interior halo ----
__global__
void build_halo_interior_kernel(
    double* __restrict__ d_u_halo,
    const double* __restrict__ d_u,
    int nx, int ny,
    int halo_nx, int halo_ny)
{
    int ii = blockIdx.x * blockDim.x + threadIdx.x;
    int jj = blockIdx.y * blockDim.y + threadIdx.y;

    if (ii < nx && jj < ny) {
        int interior_id = jj * nx + ii;
        int halo_id     = (jj + 1) * halo_nx + (ii + 1);
        d_u_halo[halo_id] = d_u[interior_id];
    }
}

void build_halo_interior_cuda(
    double* d_u_halo,
    const double* d_u,
    int nx, int ny,
    int halo_nx, int halo_ny,
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y);

    build_halo_interior_kernel<<<grid, block, 0, stream>>>(
        d_u_halo, d_u,
        nx, ny,
        halo_nx, halo_ny);
}


// ============================================================================
// Part 4 – apply_A (5-point stencil)
// ============================================================================

__global__
void apply_A_kernel(
    int nx, int ny, int halo_nx,
    const double* __restrict__ u_halo,
    double* __restrict__ Au,double coeff)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nx * ny;
    if (idx >= total) return;

    int ii = idx % nx;
    int jj = idx / nx;

    int ih = ii + 1;
    int jh = jj + 1;

    int center = jh * halo_nx + ih;

    double uc = u_halo[center];
    double uL = u_halo[center - 1];
    double uR = u_halo[center + 1];
    double uB = u_halo[center - halo_nx];
    double uT = u_halo[center + halo_nx];

    Au[idx] = 4.0 * uc - (uL + uR + uB + uT);
}

void apply_A_cuda_device(
    int nx, int ny, int halo_nx,
    const double* d_u_halo,
    double* d_Ap,double coeff, 
    cudaStream_t stream)
{
    int total = nx * ny;
    int block = 256;
    int grid  = (total + block - 1) / block;

    apply_A_kernel<<<grid, block, 0, stream>>>(
        nx, ny, halo_nx,
        d_u_halo,
        d_Ap,
        coeff);
}


// ================================================================
// END OF FILE
// ================================================================
