#include "solver_mpi_GPU.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>

using std::vector;
using std::string;

// =======================================================
// Forward declarations: GPU kernels (.cu)
// =======================================================
void apply_A_cuda_device(int nx, int ny, int halo_nx,
                         const double* d_u_halo,
                         double* d_Ap,
                         double coeff,  // 新增参数
                         cudaStream_t stream);

double dot_cuda_device(const double* d_a, const double* d_b,
                       int n, cudaStream_t stream);

void axpy_cuda(double* d_y, const double* d_x,
               double a, int n, cudaStream_t stream);

void scale_cuda(double* d_x, double a, int n, cudaStream_t stream);

void diag_solve_cuda(double* d_z, const double* d_r,
                     const double* d_D, int n, cudaStream_t stream);

void vec_copy_cuda(double* d_y, const double* d_x,
                   int n, cudaStream_t stream);

void build_halo_interior_cuda(double* d_u_halo, const double* d_u,
                              int nx, int ny, int halo_nx, int halo_ny,
                              cudaStream_t stream);

void pack_halo_cuda(const double* d_u, int nx, int ny,
                    double* d_left, double* d_right,
                    double* d_bottom, double* d_top,
                    cudaStream_t stream);

void unpack_halo_cuda(double* d_u_halo,
                      const double* d_left, const double* d_right,
                      const double* d_bottom, const double* d_top,
                      int nx, int ny, int halo_nx, int halo_ny,
                      cudaStream_t stream);

// =======================================================
// 系数矩阵与源项构建
// =======================================================
// void MPISolver2D::build_ab_F_D()
// {
//     int Kloc = nx_ * ny_;
//     F_.assign(Kloc, 0.0);
//     Ddiag_.assign(Kloc, 0.0);
    
//     // 计算对角系数
//     double inv_h2 = 1.0 / (h_ * h_);
//     double h_squared = h_ * h_;
    
//     for (int jj = 0; jj < ny_; ++jj) {
//         int j = sub_.j0 + jj;
//         for (int ii = 0; ii < nx_; ++ii) {
//             int i = sub_.i0 + ii;
            
//             double x = A1_ + i * h1_;
//             double y = A2_ + j * h2_;
            
//             double k = 1.0;
//             if (x > 0.0 && y > 0.0) {
//                 k = 1.0 / eps_;
//             }
            
//             int id = id_interior(ii, jj);
            
//             // 右端项：原始代码中是(k == 1.0) ? 1.0 : 0.0
//             // 对于泊松方程离散，右端项需要乘h²
//             F_[id] = (k == 1.0) ? h_squared : 0.0;
            
//             // 对角预条件元：标准五点差分对角是4/h²
//             Ddiag_[id] = 4.0 * inv_h2;
//         }
//     }
// }
void MPISolver2D::build_ab_F_D() {
    int Kloc = nx_ * ny_;
    F_.assign(Kloc, 0.0);
    Ddiag_.assign(Kloc, 0.0);
    
    // 调试输出
    if (rank_ == 0) {
        std::cout << "\n=== build_ab_F_D ===\n";
        std::cout << "M_=" << M_ << ", N_=" << N_ << "\n";
        std::cout << "A1_=" << A1_ << ", B1_=" << B1_ << ", h1_=" << h1_ << "\n";
        std::cout << "A2_=" << A2_ << ", B2_=" << B2_ << ", h2_=" << h2_ << "\n";
        std::cout << "h_=" << h_ << ", eps_=" << eps_ << "\n";
        std::cout << "inv_h2_=" << inv_h2_ << "\n";
    }
    
    double h_squared = h_ * h_;
    double diag_coeff = 4.0 * inv_h2_;  // 4/h²
    
    if (rank_ == 0) {
        std::cout << "h_squared = " << h_squared << "\n";
        std::cout << "diag_coeff = " << diag_coeff << "\n";
    }
    
    // 统计信息
    double sum_F = 0.0;
    int non_zero_count = 0;
    
    for (int jj = 0; jj < ny_; ++jj) {
        int j = sub_.j0 + jj;
        for (int ii = 0; ii < nx_; ++ii) {
            int i = sub_.i0 + ii;
            double x = A1_ + i * h1_;
            double y = A2_ + j * h2_;
            
            int id = id_interior(ii, jj);
            
            // 问题可能在这里：条件判断错误
            // 原逻辑：x>0且y>0时F=0，否则F=h_squared
            // 但你的计算域是[0,1]x[0,1]，当x=0或y=0时，这个条件会怎样？
            
            // 简化：所有点都设为1.0（测试用）
            F_[id] = 1.0;  // 先设为1，不乘h_squared
            
            sum_F += F_[id];
            if (F_[id] != 0.0) non_zero_count++;
            
            Ddiag_[id] = diag_coeff;
        }
    }
    
    if (rank_ == 0) {
        std::cout << "F_ statistics:\n";
        std::cout << "  sum(F_) = " << sum_F << "\n";
        std::cout << "  average(F_) = " << (sum_F / Kloc) << "\n";
        std::cout << "  non-zero points = " << non_zero_count 
                  << " out of " << Kloc << " (" 
                  << (100.0 * non_zero_count / Kloc) << "%)\n";
        
        // 验证几个点的值
        std::cout << "Sample F values:\n";
        for (int sample = 0; sample < 5; ++sample) {
            int ii = sample * nx_ / 5;
            int jj = sample * ny_ / 5;
            int id = id_interior(ii, jj);
            double x = A1_ + (sub_.i0 + ii) * h1_;
            double y = A2_ + (sub_.j0 + jj) * h2_;
            std::cout << "  (" << x << ", " << y << ") -> F=" << F_[id] << "\n";
        }
    }
}
// =======================================================
// 构造 / 析构
// =======================================================
MPISolver2D::MPISolver2D(int M, int N, MPI_Comm comm,
                         const DomainDecomposer& decomp)
    : M_(M), N_(N), comm_(comm),
      A1_(0.0), B1_(1.0), A2_(0.0), B2_(1.0),  // 初始化成员变量
      h1_(0.0), h2_(0.0), h_(0.0), eps_(0.0), inv_h2_(0.0)
{
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &size_);

    sub_ = decomp.subdomain_for_rank(rank_);
    nx_  = sub_.nx();
    ny_  = sub_.ny();

    halo_nx_ = nx_ + 2;
    halo_ny_ = ny_ + 2;

    A1_ = 0.0;
    B1_ = 1.0;
    A2_ = 0.0;
    B2_ = 1.0;
    h1_ = (B1_ - A1_) / M_;
    h2_ = (B2_ - A2_) / N_;
    h_  = std::max(h1_, h2_);
    eps_ = h_ * h_;
    inv_h2_ = 1.0 / (h_ * h_);  // 现在可以正确访问

    setup_neighbors(decomp);
    gpu_allocated_ = false;

    // 创建CUDA stream
    cudaStreamCreate(&stream_);

    stats_.M = M_;
    stats_.N = N_;
    stats_.P = size_;
}

MPISolver2D::~MPISolver2D()
{
    free_gpu_vectors();
    cudaStreamDestroy(stream_);
}

// =======================================================
// 邻居关系设置
// =======================================================
void MPISolver2D::setup_neighbors(const DomainDecomposer& decomp)
{
    int Px = decomp.Px();
    int Py = decomp.Py();

    int ix = sub_.px;
    int iy = sub_.py;

    // 按 Px x Py 的笛卡尔拓扑来确定四个方向的邻居
    nbr_left_   = (ix > 0)    ?  iy      * Px + (ix - 1) : MPI_PROC_NULL;
    nbr_right_  = (ix < Px-1) ?  iy      * Px + (ix + 1) : MPI_PROC_NULL;
    nbr_bottom_ = (iy > 0)    ? (iy - 1) * Px + ix       : MPI_PROC_NULL;
    nbr_top_    = (iy < Py-1) ? (iy + 1) * Px + ix       : MPI_PROC_NULL;
}

// =======================================================
// GPU 内存分配 / 释放
// =======================================================
void MPISolver2D::allocate_gpu_vectors()
{
    if (gpu_allocated_) return;

    int Kloc = nx_ * ny_;
    int Hloc = halo_nx_ * halo_ny_;

    cudaMalloc(&d_w_,    Kloc * sizeof(double));
    cudaMalloc(&d_r_,    Kloc * sizeof(double));
    cudaMalloc(&d_p_,    Kloc * sizeof(double));
    cudaMalloc(&d_Ap_,   Kloc * sizeof(double));
    cudaMalloc(&d_z_,    Kloc * sizeof(double));
    cudaMalloc(&d_znew_, Kloc * sizeof(double));
    cudaMalloc(&d_F_,    Kloc * sizeof(double));
    cudaMalloc(&d_D_,    Kloc * sizeof(double));
    cudaMalloc(&d_u_halo_, Hloc * sizeof(double));

    cudaMalloc(&d_left_send_,   ny_ * sizeof(double));
    cudaMalloc(&d_right_send_,  ny_ * sizeof(double));
    cudaMalloc(&d_bottom_send_, nx_ * sizeof(double));
    cudaMalloc(&d_top_send_,    nx_ * sizeof(double));
    cudaMalloc(&d_left_recv_,   ny_ * sizeof(double));
    cudaMalloc(&d_right_recv_,  ny_ * sizeof(double));
    cudaMalloc(&d_bottom_recv_, nx_ * sizeof(double));
    cudaMalloc(&d_top_recv_,    nx_ * sizeof(double));

    gpu_allocated_ = true;
}

void MPISolver2D::free_gpu_vectors()
{
    if (!gpu_allocated_) return;

    cudaFree(d_w_); cudaFree(d_r_); cudaFree(d_p_);
    cudaFree(d_Ap_); cudaFree(d_z_); cudaFree(d_znew_);
    cudaFree(d_F_); cudaFree(d_D_); cudaFree(d_u_halo_);
    cudaFree(d_left_send_); cudaFree(d_right_send_);
    cudaFree(d_bottom_send_); cudaFree(d_top_send_);
    cudaFree(d_left_recv_);  cudaFree(d_right_recv_);
    cudaFree(d_bottom_recv_); cudaFree(d_top_recv_);
    gpu_allocated_ = false;
}

// =======================================================
// Halo 交换
// =======================================================
void MPISolver2D::exchange_halo_gpu(const double* d_u)
{
    if (!gpu_allocated_) allocate_gpu_vectors();

    // Step 1: GPU pack halo
    pack_halo_cuda(d_u, nx_, ny_,
                   d_left_send_, d_right_send_,
                   d_bottom_send_, d_top_send_,
                   stream_);

    // Step 2: GPU→CPU async memcpy
    host_left_send_.resize(ny_);
    host_right_send_.resize(ny_);
    host_bottom_send_.resize(nx_);
    host_top_send_.resize(nx_);

    cudaMemcpyAsync(host_left_send_.data(), d_left_send_, ny_ * sizeof(double), cudaMemcpyDeviceToHost, stream_);
    cudaMemcpyAsync(host_right_send_.data(), d_right_send_, ny_ * sizeof(double), cudaMemcpyDeviceToHost, stream_);
    cudaMemcpyAsync(host_bottom_send_.data(), d_bottom_send_, nx_ * sizeof(double), cudaMemcpyDeviceToHost, stream_);
    cudaMemcpyAsync(host_top_send_.data(), d_top_send_, nx_ * sizeof(double), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    // Step 3: MPI 交换
    host_left_recv_.resize(ny_);
    host_right_recv_.resize(ny_);
    host_bottom_recv_.resize(nx_);
    host_top_recv_.resize(nx_);

    if (nbr_left_ != MPI_PROC_NULL)
        MPI_Sendrecv(host_left_send_.data(), ny_, MPI_DOUBLE, nbr_left_, 0,
                     host_left_recv_.data(), ny_, MPI_DOUBLE, nbr_left_, 0,
                     comm_, MPI_STATUS_IGNORE);
    else std::fill(host_left_recv_.begin(), host_left_recv_.end(), 0.0);

    if (nbr_right_ != MPI_PROC_NULL)
        MPI_Sendrecv(host_right_send_.data(), ny_, MPI_DOUBLE, nbr_right_, 1,
                     host_right_recv_.data(), ny_, MPI_DOUBLE, nbr_right_, 1,
                     comm_, MPI_STATUS_IGNORE);
    else std::fill(host_right_recv_.begin(), host_right_recv_.end(), 0.0);

    if (nbr_bottom_ != MPI_PROC_NULL)
        MPI_Sendrecv(host_bottom_send_.data(), nx_, MPI_DOUBLE, nbr_bottom_, 2,
                     host_bottom_recv_.data(), nx_, MPI_DOUBLE, nbr_bottom_, 2,
                     comm_, MPI_STATUS_IGNORE);
    else std::fill(host_bottom_recv_.begin(), host_bottom_recv_.end(), 0.0);

    if (nbr_top_ != MPI_PROC_NULL)
        MPI_Sendrecv(host_top_send_.data(), nx_, MPI_DOUBLE, nbr_top_, 3,
                     host_top_recv_.data(), nx_, MPI_DOUBLE, nbr_top_, 3,
                     comm_, MPI_STATUS_IGNORE);
    else std::fill(host_top_recv_.begin(), host_top_recv_.end(), 0.0);

    // Step 4: CPU→GPU async memcpy
    cudaMemcpyAsync(d_left_recv_, host_left_recv_.data(), ny_ * sizeof(double), cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_right_recv_, host_right_recv_.data(), ny_ * sizeof(double), cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_bottom_recv_, host_bottom_recv_.data(), nx_ * sizeof(double), cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_top_recv_, host_top_recv_.data(), nx_ * sizeof(double), cudaMemcpyHostToDevice, stream_);

    // Step 5: GPU unpack halo
    unpack_halo_cuda(d_u_halo_,
                     d_left_recv_, d_right_recv_,
                     d_bottom_recv_, d_top_recv_,
                     nx_, ny_, halo_nx_, halo_ny_, stream_);

    // Step 6: 内部 halo build
    build_halo_interior_cuda(d_u_halo_, d_u, nx_, ny_, halo_nx_, halo_ny_, stream_);

    cudaStreamSynchronize(stream_);
}

// =======================================================
// apply_A_GPU
// =======================================================
void MPISolver2D::apply_A_GPU(const double* d_u, double* d_Au)
{
    exchange_halo_gpu(d_u);
    apply_A_cuda_device(nx_, ny_, halo_nx_, d_u_halo_, d_Au, inv_h2_, stream_);
    cudaStreamSynchronize(stream_);
}

// =======================================================
// PCG求解
// =======================================================
// int MPISolver2D::pcg(double tol, double rtol, int maxIt, bool verbose)
// {
//     const int Kloc = nx_ * ny_;
//     if (!gpu_allocated_) allocate_gpu_vectors();

//     // 将数据拷贝到GPU
//     cudaMemcpyAsync(d_F_, F_.data(), Kloc * sizeof(double), cudaMemcpyHostToDevice, stream_);
//     cudaMemcpyAsync(d_D_, Ddiag_.data(), Kloc * sizeof(double), cudaMemcpyHostToDevice, stream_);
//     cudaStreamSynchronize(stream_);

//     // 初始化
//     cudaMemsetAsync(d_w_, 0, Kloc * sizeof(double), stream_);
//     vec_copy_cuda(d_r_, d_F_, Kloc, stream_);
//     diag_solve_cuda(d_z_, d_r_, d_D_, Kloc, stream_);
//     vec_copy_cuda(d_p_, d_z_, Kloc, stream_);
//     cudaStreamSynchronize(stream_);

//     // 计算初始内积
//     double zr_loc = dot_cuda_device(d_z_, d_r_, Kloc, stream_);
//     double zr_glob = 0.0;
//     MPI_Allreduce(&zr_loc, &zr_glob, 1, MPI_DOUBLE, MPI_SUM, comm_);

//     double rr_loc = dot_cuda_device(d_r_, d_r_, Kloc, stream_);
//     double rr_glob = 0.0;
//     MPI_Allreduce(&rr_loc, &rr_glob, 1, MPI_DOUBLE, MPI_SUM, comm_);
//     double normF = std::sqrt(rr_glob * h1_ * h2_);
//     if (normF < 1e-3) normF = 1.0;

//     int it = 0;
//     for (; it < maxIt; ++it) {
//         // 计算Ap
//         apply_A_GPU(d_p_, d_Ap_);

//         // 计算pAp
//         double pAp_loc = dot_cuda_device(d_p_, d_Ap_, Kloc, stream_);
//         double pAp_glob = 0.0;
//         MPI_Allreduce(&pAp_loc, &pAp_glob, 1, MPI_DOUBLE, MPI_SUM, comm_);
//         double alpha = zr_glob / pAp_glob;

//         // 更新解和残差
//         axpy_cuda(d_w_, d_p_, alpha, Kloc, stream_);
//         axpy_cuda(d_r_, d_Ap_, -alpha, Kloc, stream_);
//         cudaStreamSynchronize(stream_);

//         // 检查收敛
//         double rr_loc2 = dot_cuda_device(d_r_, d_r_, Kloc, stream_);
//         double rr_glob2 = 0.0;
//         MPI_Allreduce(&rr_loc2, &rr_glob2, 1, MPI_DOUBLE, MPI_SUM, comm_);
//         double norm_r = std::sqrt(rr_glob2 * h1_ * h2_);
        
//         // if (verbose && rank_ == 0)
//         if (it % 1000 && rank_ == 0)
//             std::cout << "[PCG] iter=" << it << "  ||r||_E=" << norm_r 
//                       << "  rel=" << norm_r/normF << std::endl;
        
//         if (norm_r < tol || norm_r / normF < rtol) { 
//             ++it; 
//             break; 
//         }

//         // 预条件
//         diag_solve_cuda(d_znew_, d_r_, d_D_, Kloc, stream_);
//         cudaStreamSynchronize(stream_);

//         // 计算新的zr
//         double zr_new_loc = dot_cuda_device(d_znew_, d_r_, Kloc, stream_);
//         double zr_new_glob = 0.0;
//         MPI_Allreduce(&zr_new_loc, &zr_new_glob, 1, MPI_DOUBLE, MPI_SUM, comm_);
//         double beta = zr_new_glob / zr_glob;

//         // 更新搜索方向
//         scale_cuda(d_p_, beta, Kloc, stream_);
//         axpy_cuda(d_p_, d_znew_, 1.0, Kloc, stream_);
//         vec_copy_cuda(d_z_, d_znew_, Kloc, stream_);
//         zr_glob = zr_new_glob;
//     }

//     // 将结果拷贝回CPU
//     w_.resize(Kloc);
//     cudaMemcpyAsync(w_.data(), d_w_, Kloc * sizeof(double), cudaMemcpyDeviceToHost, stream_);
//     cudaStreamSynchronize(stream_);

//     stats_.iters = it;
//     return it;
// }
// int MPISolver2D::pcg(double tol, double rtol, int maxIt, bool verbose)
// {
//     const int Kloc = nx_ * ny_;
//     if (!gpu_allocated_) allocate_gpu_vectors();
    
//     // 调试：打印关键参数
//     if (verbose && rank_ == 0) {
//         std::cout << "[DEBUG] PCG parameters:\n";
//         std::cout << "  h1=" << h1_ << ", h2=" << h2_ << ", h=" << h_ << "\n";
//         std::cout << "  inv_h2=" << inv_h2_ << "\n";
//         std::cout << "  nx=" << nx_ << ", ny=" << ny_ << ", Kloc=" << Kloc << "\n";
//     }
    
//     // 将数据拷贝到GPU
//     cudaMemcpyAsync(d_F_, F_.data(), Kloc * sizeof(double), cudaMemcpyHostToDevice, stream_);
//     cudaMemcpyAsync(d_D_, Ddiag_.data(), Kloc * sizeof(double), cudaMemcpyHostToDevice, stream_);
//     cudaStreamSynchronize(stream_);
    
//     // 调试：检查F_和Ddiag_的值
//     if (verbose && rank_ == 0) {
//         std::vector<double> cpu_F(Kloc), cpu_D(Kloc);
//         cudaMemcpy(cpu_F.data(), d_F_, Kloc * sizeof(double), cudaMemcpyDeviceToHost);
//         cudaMemcpy(cpu_D.data(), d_D_, Kloc * sizeof(double), cudaMemcpyDeviceToHost);
//         double max_F = *std::max_element(cpu_F.begin(), cpu_F.end());
//         double min_F = *std::min_element(cpu_F.begin(), cpu_F.end());
//         double max_D = *std::max_element(cpu_D.begin(), cpu_D.end());
//         double min_D = *std::min_element(cpu_D.begin(), cpu_D.end());
//         std::cout << "[DEBUG] F range: " << min_F << " to " << max_F << "\n";
//         std::cout << "[DEBUG] D range: " << min_D << " to " << max_D << "\n";
//     }
    
//     // 初始化
//     cudaMemsetAsync(d_w_, 0, Kloc * sizeof(double), stream_);
//     vec_copy_cuda(d_r_, d_F_, Kloc, stream_);
//     diag_solve_cuda(d_z_, d_r_, d_D_, Kloc, stream_);
//     vec_copy_cuda(d_p_, d_z_, Kloc, stream_);
//     cudaStreamSynchronize(stream_);
    
//     // 计算初始内积
//     double zr_loc = dot_cuda_device(d_z_, d_r_, Kloc, stream_);
//     double zr_glob = 0.0;
//     MPI_Allreduce(&zr_loc, &zr_glob, 1, MPI_DOUBLE, MPI_SUM, comm_);
    
//     double rr_loc = dot_cuda_device(d_r_, d_r_, Kloc, stream_);
//     double rr_glob = 0.0;
//     MPI_Allreduce(&rr_loc, &rr_glob, 1, MPI_DOUBLE, MPI_SUM, comm_);
//     double normF = std::sqrt(rr_glob * h1_ * h2_);
//     if (normF < 1e-10) normF = 1.0;
    
//     // 调试：检查初始值
//     if (verbose && rank_ == 0) {
//         std::cout << "[DEBUG] Initial: zr_glob=" << zr_glob 
//                   << ", rr_glob=" << rr_glob 
//                   << ", normF=" << normF << "\n";
//     }
    
//     int it = 0;
//     for (; it < maxIt; ++it) {
//         // 计算Ap
//         apply_A_GPU(d_p_, d_Ap_);
        
//         // 计算pAp
//         double pAp_loc = dot_cuda_device(d_p_, d_Ap_, Kloc, stream_);
//         double pAp_glob = 0.0;
//         MPI_Allreduce(&pAp_loc, &pAp_glob, 1, MPI_DOUBLE, MPI_SUM, comm_);
        
//         // 调试：检查pAp
//         if (std::isnan(pAp_glob) || std::abs(pAp_glob) < 1e-15) {
//             if (verbose && rank_ == 0) {
//                 std::cout << "[WARNING] pAp_glob = " << pAp_glob << " at iter " << it << "\n";
//             }
//             break;
//         }
        
//         double alpha = zr_glob / pAp_glob;
        
//         // 更新解和残差
//         axpy_cuda(d_w_, d_p_, alpha, Kloc, stream_);
//         axpy_cuda(d_r_, d_Ap_, -alpha, Kloc, stream_);
//         cudaStreamSynchronize(stream_);
        
//         // 检查收敛
//         double rr_loc2 = dot_cuda_device(d_r_, d_r_, Kloc, stream_);
//         double rr_glob2 = 0.0;
//         MPI_Allreduce(&rr_loc2, &rr_glob2, 1, MPI_DOUBLE, MPI_SUM, comm_);
//         double norm_r = std::sqrt(rr_glob2 * h1_ * h2_);
        
//         // 调试：每100次迭代输出一次
//         if (verbose && rank_ == 0 && (it % 100 == 0 || it < 10)) {
//             std::cout << "[PCG] iter=" << it 
//                       << " ||r||=" << norm_r 
//                       << " rel=" << norm_r/normF
//                       << " alpha=" << alpha 
//                       << " pAp=" << pAp_glob << "\n";
//         }
        
//         if (norm_r < tol || norm_r / normF < rtol) { 
//             ++it; 
//             if (verbose && rank_ == 0) {
//                 std::cout << "[PCG] Converged at iter=" << it << "\n";
//             }
//             break; 
//         }
        
//         // 预条件
//         diag_solve_cuda(d_znew_, d_r_, d_D_, Kloc, stream_);
//         cudaStreamSynchronize(stream_);
        
//         // 计算新的zr
//         double zr_new_loc = dot_cuda_device(d_znew_, d_r_, Kloc, stream_);
//         double zr_new_glob = 0.0;
//         MPI_Allreduce(&zr_new_loc, &zr_new_glob, 1, MPI_DOUBLE, MPI_SUM, comm_);
        
//         // 检查zr_glob是否为0
//         if (std::abs(zr_glob) < 1e-15) {
//             if (verbose && rank_ == 0) {
//                 std::cout << "[WARNING] zr_glob too small: " << zr_glob << "\n";
//             }
//             break;
//         }
        
//         double beta = zr_new_glob / zr_glob;
        
//         // 更新搜索方向
//         scale_cuda(d_p_, beta, Kloc, stream_);
//         axpy_cuda(d_p_, d_znew_, 1.0, Kloc, stream_);
//         vec_copy_cuda(d_z_, d_znew_, Kloc, stream_);
//         zr_glob = zr_new_glob;
//     }
    
//     // 将结果拷贝回CPU
//     w_.resize(Kloc);
//     cudaMemcpyAsync(w_.data(), d_w_, Kloc * sizeof(double), cudaMemcpyDeviceToHost, stream_);
//     cudaStreamSynchronize(stream_);
    
//     stats_.iters = it;
//     return it;
// }
// int MPISolver2D::pcg(double tol, double rtol, int maxIt, bool verbose)
// {
//     const int Kloc = nx_ * ny_;
//     if (!gpu_allocated_) allocate_gpu_vectors();
    
//     // 强制启用verbose输出
//     verbose = true;
    
//     if (verbose && rank_ == 0) {
//         std::cout << "\n=== PCG DEBUG INFO ===\n";
//         std::cout << "tol = " << tol << ", rtol = " << rtol << "\n";
//         std::cout << "h1 = " << h1_ << ", h2 = " << h2_ << ", h = " << h_ << "\n";
//         std::cout << "inv_h2 = " << inv_h2_ << "\n";
//         std::cout << "nx = " << nx_ << ", ny = " << ny_ << ", Kloc = " << Kloc << "\n";
//     }
    
//     // 将数据拷贝到GPU
//     cudaMemcpyAsync(d_F_, F_.data(), Kloc * sizeof(double), cudaMemcpyHostToDevice, stream_);
//     cudaMemcpyAsync(d_D_, Ddiag_.data(), Kloc * sizeof(double), cudaMemcpyHostToDevice, stream_);
//     cudaStreamSynchronize(stream_);
    
//     // 初始化
//     cudaMemsetAsync(d_w_, 0, Kloc * sizeof(double), stream_);
//     vec_copy_cuda(d_r_, d_F_, Kloc, stream_);
//     diag_solve_cuda(d_z_, d_r_, d_D_, Kloc, stream_);
//     vec_copy_cuda(d_p_, d_z_, Kloc, stream_);
//     cudaStreamSynchronize(stream_);
    
//     // 计算初始内积
//     double zr_loc = dot_cuda_device(d_z_, d_r_, Kloc, stream_);
//     double zr_glob = 0.0;
//     MPI_Allreduce(&zr_loc, &zr_glob, 1, MPI_DOUBLE, MPI_SUM, comm_);
    
//     double rr_loc = dot_cuda_device(d_r_, d_r_, Kloc, stream_);
//     double rr_glob = 0.0;
//     MPI_Allreduce(&rr_loc, &rr_glob, 1, MPI_DOUBLE, MPI_SUM, comm_);
    
//     double normF = std::sqrt(rr_glob * h1_ * h2_);
//     double norm_r = std::sqrt(rr_glob * h1_ * h2_);
    
//     if (verbose && rank_ == 0) {
//         std::cout << "\nInitial residual check:\n";
//         std::cout << "rr_glob = " << rr_glob << "\n";
//         std::cout << "normF = " << normF << "\n";
//         std::cout << "norm_r = " << norm_r << "\n";
//         std::cout << "zr_glob = " << zr_glob << "\n";
//         std::cout << "tol = " << tol << "\n";
//         std::cout << "rtol * normF = " << rtol * normF << "\n";
//         std::cout << "Check: norm_r < tol? " << (norm_r < tol) << "\n";
//         std::cout << "Check: norm_r/normF < rtol? " << (norm_r/normF < rtol) << "\n";
//     }
    
//     // 如果初始残差已经很小，直接返回
//     if (norm_r < tol || norm_r / normF < rtol) { 
//         if (verbose && rank_ == 0) {
//             std::cout << "\nWARNING: Initial residual already below tolerance!\n";
//             std::cout << "This usually means F_ (right-hand side) is too small or zero.\n";
//         }
        
//         // 将结果拷贝回CPU
//         w_.resize(Kloc);
//         cudaMemcpyAsync(w_.data(), d_w_, Kloc * sizeof(double), cudaMemcpyDeviceToHost, stream_);
//         cudaStreamSynchronize(stream_);
        
//         stats_.iters = 0;
//         return 0;
//     }
    
//     if (verbose && rank_ == 0) {
//         std::cout << "\nStarting PCG iterations...\n";
//     }
    
//     int it = 0;
//     for (; it < maxIt; ++it) {
//         // 计算Ap
//         apply_A_GPU(d_p_, d_Ap_);
        
//         // 计算pAp
//         double pAp_loc = dot_cuda_device(d_p_, d_Ap_, Kloc, stream_);
//         double pAp_glob = 0.0;
//         MPI_Allreduce(&pAp_loc, &pAp_glob, 1, MPI_DOUBLE, MPI_SUM, comm_);
        
//         if (verbose && rank_ == 0 && it == 0) {
//             std::cout << "\nFirst iteration details:\n";
//             std::cout << "pAp_glob = " << pAp_glob << "\n";
//             std::cout << "zr_glob = " << zr_glob << "\n";
//         }
        
//         // 安全检查：避免除零
//         if (std::abs(pAp_glob) < 1e-15) {
//             if (verbose && rank_ == 0) {
//                 std::cout << "\nERROR: pAp_glob too small: " << pAp_glob << "\n";
//             }
//             break;
//         }
        
//         double alpha = zr_glob / pAp_glob;
        
//         if (verbose && rank_ == 0 && it == 0) {
//             std::cout << "alpha = " << alpha << "\n";
//         }
        
//         // 更新解和残差
//         axpy_cuda(d_w_, d_p_, alpha, Kloc, stream_);
//         axpy_cuda(d_r_, d_Ap_, -alpha, Kloc, stream_);
//         cudaStreamSynchronize(stream_);
        
//         // 检查收敛
//         double rr_loc2 = dot_cuda_device(d_r_, d_r_, Kloc, stream_);
//         double rr_glob2 = 0.0;
//         MPI_Allreduce(&rr_loc2, &rr_glob2, 1, MPI_DOUBLE, MPI_SUM, comm_);
//         double norm_r = std::sqrt(rr_glob2 * h1_ * h2_);
        
//         if (verbose && rank_ == 0 && (it < 5 || it % 100 == 0)) {
//             std::cout << "[PCG] iter=" << it 
//                       << " ||r||=" << norm_r 
//                       << " rel=" << norm_r/normF
//                       << " alpha=" << alpha << "\n";
//         }
        
//         if (norm_r < tol || norm_r / normF < rtol) { 
//             ++it; 
//             if (verbose && rank_ == 0) {
//                 std::cout << "\nConverged at iter=" << it << "\n";
//             }
//             break; 
//         }
        
//         // 预条件
//         diag_solve_cuda(d_znew_, d_r_, d_D_, Kloc, stream_);
//         cudaStreamSynchronize(stream_);
        
//         // 计算新的zr
//         double zr_new_loc = dot_cuda_device(d_znew_, d_r_, Kloc, stream_);
//         double zr_new_glob = 0.0;
//         MPI_Allreduce(&zr_new_loc, &zr_new_glob, 1, MPI_DOUBLE, MPI_SUM, comm_);
        
//         // 安全检查
//         if (std::abs(zr_glob) < 1e-15) {
//             if (verbose && rank_ == 0) {
//                 std::cout << "\nERROR: zr_glob too small: " << zr_glob << "\n";
//             }
//             break;
//         }
        
//         double beta = zr_new_glob / zr_glob;
        
//         if (verbose && rank_ == 0 && it == 0) {
//             std::cout << "zr_new_glob = " << zr_new_glob << "\n";
//             std::cout << "beta = " << beta << "\n";
//         }
        
//         // 更新搜索方向
//         scale_cuda(d_p_, beta, Kloc, stream_);
//         axpy_cuda(d_p_, d_znew_, 1.0, Kloc, stream_);
//         vec_copy_cuda(d_z_, d_znew_, Kloc, stream_);
//         zr_glob = zr_new_glob;
        
//         // 强制至少运行几次迭代以观察行为
//         if (it >= 4 && verbose && rank_ == 0) {
//             std::cout << "\nStopping after 5 iterations for debugging...\n";
//             break;
//         }
//     }
    
//     // 将结果拷贝回CPU
//     w_.resize(Kloc);
//     cudaMemcpyAsync(w_.data(), d_w_, Kloc * sizeof(double), cudaMemcpyDeviceToHost, stream_);
//     cudaStreamSynchronize(stream_);
    
//     stats_.iters = it;
//     return it;
// }
int MPISolver2D::pcg(double tol, double rtol, int maxIt, bool verbose)
{
    const int Kloc = nx_ * ny_;
    if (!gpu_allocated_) allocate_gpu_vectors();
    
    verbose = true;  // 强制详细输出
    
    if (verbose && rank_ == 0) {
        std::cout << "\n=== PCG START ===\n";
    }
    
    // 1. 将数据拷贝到GPU
    cudaMemcpyAsync(d_F_, F_.data(), Kloc * sizeof(double), cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_D_, Ddiag_.data(), Kloc * sizeof(double), cudaMemcpyHostToDevice, stream_);
    cudaStreamSynchronize(stream_);
    
    // 2. 初始化
    cudaMemsetAsync(d_w_, 0, Kloc * sizeof(double), stream_);
    vec_copy_cuda(d_r_, d_F_, Kloc, stream_);
    
    // 3. 计算初始残差范数
    double rr_loc = dot_cuda_device(d_r_, d_r_, Kloc, stream_);
    double rr_glob = 0.0;
    MPI_Allreduce(&rr_loc, &rr_glob, 1, MPI_DOUBLE, MPI_SUM, comm_);
    
    double normF = std::sqrt(rr_glob);  // 注意：不乘h1*h2！
    
    if (verbose && rank_ == 0) {
        std::cout << "\nInitial residual calculation:\n";
        std::cout << "rr_loc = " << rr_loc << "\n";
        std::cout << "rr_glob = " << rr_glob << "\n";
        std::cout << "sqrt(rr_glob) = " << normF << "\n";
        std::cout << "tol = " << tol << "\n";
        std::cout << "normF = " << normF << "\n";
        
        // 如果normF太小，手动设置一个值
        if (normF < 1e-10) {
            std::cout << "WARNING: normF too small! Setting to 1.0 for relative tolerance.\n";
            normF = 1.0;
        }
    }
    
    // 4. 如果初始残差已经很小，但仍然需要求解
    if (normF < tol) {
        if (verbose && rank_ == 0) {
            std::cout << "\nInitial residual already below absolute tolerance.\n";
            std::cout << "But continuing anyway to test the algorithm...\n";
        }
    }
    
    // 5. 继续PCG初始化
    diag_solve_cuda(d_z_, d_r_, d_D_, Kloc, stream_);
    vec_copy_cuda(d_p_, d_z_, Kloc, stream_);
    cudaStreamSynchronize(stream_);
    
    double zr_loc = dot_cuda_device(d_z_, d_r_, Kloc, stream_);
    double zr_glob = 0.0;
    MPI_Allreduce(&zr_loc, &zr_glob, 1, MPI_DOUBLE, MPI_SUM, comm_);
    
    if (verbose && rank_ == 0) {
        std::cout << "\nPCG initialization:\n";
        std::cout << "zr_glob = " << zr_glob << "\n";
        std::cout << "Starting iterations...\n";
    }
    
    int it = 0;
    for (; it < maxIt; ++it) {
        // 计算Ap
        apply_A_GPU(d_p_, d_Ap_);
        
        // 计算pAp
        double pAp_loc = dot_cuda_device(d_p_, d_Ap_, Kloc, stream_);
        double pAp_glob = 0.0;
        MPI_Allreduce(&pAp_loc, &pAp_glob, 1, MPI_DOUBLE, MPI_SUM, comm_);
        
        if (std::abs(pAp_glob) < 1e-15) {
            if (verbose && rank_ == 0) {
                std::cout << "ERROR: pAp_glob = " << pAp_glob << " (too small)\n";
            }
            break;
        }
        
        double alpha = zr_glob / pAp_glob;
        
        // 更新解和残差
        axpy_cuda(d_w_, d_p_, alpha, Kloc, stream_);
        axpy_cuda(d_r_, d_Ap_, -alpha, Kloc, stream_);
        cudaStreamSynchronize(stream_);
        
        // 检查收敛
        double rr_loc2 = dot_cuda_device(d_r_, d_r_, Kloc, stream_);
        double rr_glob2 = 0.0;
        MPI_Allreduce(&rr_loc2, &rr_glob2, 1, MPI_DOUBLE, MPI_SUM, comm_);
        double norm_r = std::sqrt(rr_glob2);
        
        if (verbose && rank_ == 0 && (it < 10 || it % 100 == 0)) {
            // std::cout << std::scientific << std::setprecision(6);
            std::cout << "[PCG] iter=" << it 
                      << " ||r||=" << norm_r 
                      << " rel=" << norm_r/normF
                      << " alpha=" << alpha 
                      << " pAp=" << pAp_glob << "\n";
        }
        
        // 仅使用相对容差判断
        if (norm_r / normF < rtol) { 
            ++it; 
            if (verbose && rank_ == 0) {
                std::cout << "\nConverged! iter=" << it 
                          << ", ||r||/||F|| = " << (norm_r/normF) 
                          << " < " << rtol << "\n";
            }
            break; 
        }
        
        // 预条件
        diag_solve_cuda(d_znew_, d_r_, d_D_, Kloc, stream_);
        cudaStreamSynchronize(stream_);
        
        // 计算新的zr
        double zr_new_loc = dot_cuda_device(d_znew_, d_r_, Kloc, stream_);
        double zr_new_glob = 0.0;
        MPI_Allreduce(&zr_new_loc, &zr_new_glob, 1, MPI_DOUBLE, MPI_SUM, comm_);
        
        double beta = zr_new_glob / zr_glob;
        
        // 更新搜索方向
        scale_cuda(d_p_, beta, Kloc, stream_);
        axpy_cuda(d_p_, d_znew_, 1.0, Kloc, stream_);
        vec_copy_cuda(d_z_, d_znew_, Kloc, stream_);
        zr_glob = zr_new_glob;
    }
    
    // 将结果拷贝回CPU
    w_.resize(Kloc);
    cudaMemcpyAsync(w_.data(), d_w_, Kloc * sizeof(double), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
    
    stats_.iters = it;
    
    if (verbose && rank_ == 0) {
        std::cout << "\n=== PCG FINISHED ===\n";
        std::cout << "Total iterations: " << it << "\n";
    }
    
    return it;
}
// =======================================================
// 收集并保存解
// =======================================================
void MPISolver2D::gather_and_save(const std::string& fname) const
{
    int Kloc = nx_ * ny_;
    std::vector<double> loc(Kloc);
    for (int jj = 0; jj < ny_; ++jj)
        for (int ii = 0; ii < nx_; ++ii)
            loc[id_interior(ii, jj)] = w_[id_interior(ii, jj)];

    int locCount = Kloc;
    std::vector<int> counts(size_), displs(size_);
    MPI_Gather(&locCount, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, comm_);
    if (rank_ == 0) {
        displs[0] = 0;
        for (int i = 1; i < size_; ++i)
            displs[i] = displs[i - 1] + counts[i - 1];
    }

    std::vector<double> global;
    int totalCount = 0;
    if (rank_ == 0) {
        totalCount = displs[size_ - 1] + counts[size_ - 1];
        global.resize(totalCount);
    }

    MPI_Gatherv(loc.data(), locCount, MPI_DOUBLE,
                global.data(), counts.data(), displs.data(),
                MPI_DOUBLE, 0, comm_);

    if (rank_ == 0) {
        std::ofstream f(fname);
        if (!f) { std::cerr << "[Error] Cannot open file: " << fname << "\n"; return; }
        for (double v : global) f << v << "\n";
        std::cerr << "[Info] Solution saved to " << fname << "\n";
    }
}