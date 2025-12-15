#include "solver_mpi_GPU.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>

// 核函数声明（真正的实现放在 solver_mpi_GPU_kernels.cu 里）
__global__
extern void apply_A_cuda(int nx, int ny, int halo_nx,
                         const double* d_u_halo, double* d_Au);


using std::vector;
using std::string;


// =======================================
//          构造函数 / 析构函数
// =======================================
MPISolver2D::MPISolver2D(int M, int N, MPI_Comm comm,
                         const DomainDecomposer& decomp)
    : M_(M), N_(N), comm_(comm)
{
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &size_);

    sub_ = decomp.subdomain_for_rank(rank_);
    nx_  = sub_.nx();
    ny_  = sub_.ny();
    halo_nx_ = nx_ + 2;
    halo_ny_ = ny_ + 2;

    h1_ = (B1_ - A1_) / M_;
    h2_ = (B2_ - A2_) / N_;
    h_  = std::max(h1_, h2_);
    eps_ = h_ * h_;

    setup_neighbors(decomp);

    gpu_buffers_allocated_ = false;

    stats_.M = M_;
    stats_.N = N_;
    stats_.P = size_;
}

MPISolver2D::~MPISolver2D()
{
    free_gpu_buffers();
}

// 邻居关系：按 Px x Py 上的笛卡尔排布
void MPISolver2D::setup_neighbors(const DomainDecomposer& decomp)
{
    int Px = decomp.Px();
    int Py = decomp.Py();
    int ix = sub_.px;
    int iy = sub_.py;

    nbr_left_   = (ix > 0)     ? iy * Px + (ix - 1) : MPI_PROC_NULL;
    nbr_right_  = (ix < Px-1)  ? iy * Px + (ix + 1) : MPI_PROC_NULL;
    nbr_bottom_ = (iy > 0)     ? (iy - 1) * Px + ix : MPI_PROC_NULL;
    nbr_top_    = (iy < Py-1)  ? (iy + 1) * Px + ix : MPI_PROC_NULL;
}

// =======================================
//          系数矩阵与源项构建
// =======================================
void MPISolver2D::build_ab_F_D()
{
    const int Kloc = nx_ * ny_;

    aL_.assign(Kloc, 1.0);
    aR_.assign(Kloc, 1.0);
    bB_.assign(Kloc, 1.0);
    bT_.assign(Kloc, 1.0);
    F_.assign(Kloc, 0.0);
    Ddiag_.assign(Kloc, 0.0);

    // 简化版：k(x,y) = 1 (第一象限外) 或 1/eps_（第一象限内：x>0,y>0）
    for (int jj = 0; jj < ny_; ++jj) {
        int j = sub_.j0 + jj;
        for (int ii = 0; ii < nx_; ++ii) {
            int i = sub_.i0 + ii;
            double x = A1_ + i * h1_;
            double y = A2_ + j * h2_;

            double k = 1.0;
            if (x > 0.0 && y > 0.0) k = 1.0 / eps_; // 靴子外区域

            int id = id_interior(ii, jj);
            F_[id]      = (k == 1.0) ? 1.0 : 0.0; // 右端项
            Ddiag_[id]  = 4.0;                // Jacobi 预条件对角元
        }
    }
}

// =======================================
//          Halo 交换函数（CPU + MPI）
// =======================================
// void MPISolver2D::exchange_halo(vector<double>& u_halo) const
// {
//     // u_halo 尺寸应为 halo_nx_ * halo_ny_
//     const int haloSize = halo_nx_ * halo_ny_;
//     if ((int)u_halo.size() != haloSize) {
//         // 为安全起见，直接退出（实际项目可改成 assert/异常）
//         std::cerr << "exchange_halo: invalid halo size on rank "
//                   << rank_ << std::endl;
//         MPI_Abort(comm_, -1);
//     }

//     // 左右方向
//     vector<double> sendL(ny_), recvL(ny_);
//     vector<double> sendR(ny_), recvR(ny_);

//     for (int jj = 0; jj < ny_; ++jj) {
//         // 内部列 i=1..nx_ 对应 halo i=1..nx_
//         sendL[jj] = u_halo[id_halo(1,      jj + 1)];  // 最左内部列
//         sendR[jj] = u_halo[id_halo(nx_,   jj + 1)];   // 最右内部列
//     }

//     if (nbr_left_ != MPI_PROC_NULL) {
//         MPI_Sendrecv(sendL.data(), ny_, MPI_DOUBLE, nbr_left_,  0,
//                      recvL.data(), ny_, MPI_DOUBLE, nbr_left_,  0,
//                      comm_, MPI_STATUS_IGNORE);
//         for (int jj = 0; jj < ny_; ++jj)
//             u_halo[id_halo(0, jj + 1)] = recvL[jj];
//     }

//     if (nbr_right_ != MPI_PROC_NULL) {
//         MPI_Sendrecv(sendR.data(), ny_, MPI_DOUBLE, nbr_right_, 1,
//                      recvR.data(), ny_, MPI_DOUBLE, nbr_right_, 1,
//                      comm_, MPI_STATUS_IGNORE);
//         for (int jj = 0; jj < ny_; ++jj)
//             u_halo[id_halo(nx_ + 1, jj + 1)] = recvR[jj];
//     }

//     // 上下方向
//     vector<double> sendB(nx_), recvB(nx_);
//     vector<double> sendT(nx_), recvT(nx_);

//     for (int ii = 0; ii < nx_; ++ii) {
//         sendB[ii] = u_halo[id_halo(ii + 1, 1     )]; // 最下内部行
//         sendT[ii] = u_halo[id_halo(ii + 1, ny_   )]; // 最上内部行
//     }

//     if (nbr_bottom_ != MPI_PROC_NULL) {
//         MPI_Sendrecv(sendB.data(), nx_, MPI_DOUBLE, nbr_bottom_, 2,
//                      recvB.data(), nx_, MPI_DOUBLE, nbr_bottom_, 2,
//                      comm_, MPI_STATUS_IGNORE);
//         for (int ii = 0; ii < nx_; ++ii)
//             u_halo[id_halo(ii + 1, 0)] = recvB[ii];
//     }

//     if (nbr_top_ != MPI_PROC_NULL) {
//         MPI_Sendrecv(sendT.data(), nx_, MPI_DOUBLE, nbr_top_,    3,
//                      recvT.data(), nx_, MPI_DOUBLE, nbr_top_,    3,
//                      comm_, MPI_STATUS_IGNORE);
//         for (int ii = 0; ii < nx_; ++ii)
//             u_halo[id_halo(ii + 1, ny_ + 1)] = recvT[ii];
//     }
// }
void MPISolver2D::exchange_halo(std::vector<double>& u_halo) const
{
    const int haloSize = halo_nx_ * halo_ny_;
    if ((int)u_halo.size() != haloSize) {
        std::cerr << "exchange_halo: invalid halo size on rank "
                  << rank_ << std::endl;
        MPI_Abort(comm_, -1);
    }

    // ---------- 左右方向 ----------
    std::vector<double> sendL(ny_), recvL(ny_);
    std::vector<double> sendR(ny_), recvR(ny_);

    for (int jj = 0; jj < ny_; ++jj) {
        sendL[jj] = u_halo[id_halo(1,      jj + 1)];
        sendR[jj] = u_halo[id_halo(nx_,   jj + 1)];
    }

    // 和 CPU 版一致：左 recv0/send1，右 recv1/send0
    if (nbr_left_ != MPI_PROC_NULL) {
        MPI_Sendrecv(sendL.data(), ny_, MPI_DOUBLE, nbr_left_,  1,   // send tag=1
                     recvL.data(), ny_, MPI_DOUBLE, nbr_left_,  0,   // recv tag=0
                     comm_, MPI_STATUS_IGNORE);

        for (int jj = 0; jj < ny_; ++jj)
            u_halo[id_halo(0, jj + 1)] = recvL[jj];
    }

    if (nbr_right_ != MPI_PROC_NULL) {
        MPI_Sendrecv(sendR.data(), ny_, MPI_DOUBLE, nbr_right_, 0,   // send tag=0
                     recvR.data(), ny_, MPI_DOUBLE, nbr_right_, 1,   // recv tag=1
                     comm_, MPI_STATUS_IGNORE);

        for (int jj = 0; jj < ny_; ++jj)
            u_halo[id_halo(nx_ + 1, jj + 1)] = recvR[jj];
    }

    // ---------- 上下方向 ----------
    std::vector<double> sendB(nx_), recvB(nx_);
    std::vector<double> sendT(nx_), recvT(nx_);

    for (int ii = 0; ii < nx_; ++ii) {
        sendB[ii] = u_halo[id_halo(ii + 1, 1      )];
        sendT[ii] = u_halo[id_halo(ii + 1, ny_    )];
    }

    // 和 CPU 版一致：下 recv2/send3，上 recv3/send2
    if (nbr_bottom_ != MPI_PROC_NULL) {
        MPI_Sendrecv(sendB.data(), nx_, MPI_DOUBLE, nbr_bottom_, 3,  // send tag=3
                     recvB.data(), nx_, MPI_DOUBLE, nbr_bottom_, 2,  // recv tag=2
                     comm_, MPI_STATUS_IGNORE);

        for (int ii = 0; ii < nx_; ++ii)
            u_halo[id_halo(ii + 1, 0)] = recvB[ii];
    }

    if (nbr_top_ != MPI_PROC_NULL) {
        MPI_Sendrecv(sendT.data(), nx_, MPI_DOUBLE, nbr_top_,  2,     // send tag=2
                     recvT.data(), nx_, MPI_DOUBLE, nbr_top_,  3,     // recv tag=3
                     comm_, MPI_STATUS_IGNORE);

        for (int ii = 0; ii < nx_; ++ii)
            u_halo[id_halo(ii + 1, ny_ + 1)] = recvT[ii];
    }
}


// =======================================
//          应用矩阵 A（CPU + CUDA）
// =======================================
void MPISolver2D::apply_A(const vector<double>& u, vector<double>& Au)
{
    const int Kloc = nx_ * ny_;

    // 1. 构造带 halo 的副本，做 halo 交换（在 CPU + MPI 上）
    vector<double> u_halo(halo_nx_ * halo_ny_, 0.0);
    for (int jj = 0; jj < ny_; ++jj)
        for (int ii = 0; ii < nx_; ++ii)
            u_halo[id_halo(ii + 1, jj + 1)] = u[id_interior(ii, jj)];

    exchange_halo(u_halo);

    // 2. GPU 计算 Au = A * u
    if (!gpu_buffers_allocated_) {
        allocate_gpu_buffers();
    }

    const int haloCount = halo_nx_ * halo_ny_;
    const int N = Kloc;

    cudaMemcpy(d_u_halo_, u_halo.data(),
               haloCount * sizeof(double),
               cudaMemcpyHostToDevice);

    // 这里不再直接 launch kernel，而是调用 .cu 中的封装函数
    apply_A_cuda(nx_, ny_, halo_nx_, d_u_halo_, d_Au_);

    Au.resize(N);
    cudaMemcpy(Au.data(), d_Au_,
               N * sizeof(double),
               cudaMemcpyDeviceToHost);
}


// =======================================
//          向量运算 / 范数
// =======================================
double MPISolver2D::dotE_local(const vector<double>& u,
                               const vector<double>& v) const
{
    double s = 0.0;
    const int Kloc = nx_ * ny_;
    for (int k = 0; k < Kloc; ++k)
        s += u[k] * v[k];
    return s;
}

double MPISolver2D::normE_global(const vector<double>& u) const
{
    double loc = dotE_local(u, u);
    double glob = 0.0;
    MPI_Allreduce(&loc, &glob, 1, MPI_DOUBLE, MPI_SUM, comm_);
    return std::sqrt(glob * h1_ * h2_);
}

void MPISolver2D::solve_Dz_eq_r(const vector<double>& r,
                                vector<double>& z) const
{
    const int Kloc = nx_ * ny_;
    z.assign(Kloc, 0.0);
    for (int k = 0; k < Kloc; ++k) {
        z[k] = r[k] / Ddiag_[k];
    }
}

// =======================================
//          PCG 主循环（主要在 CPU）
// =======================================
int MPISolver2D::pcg(double tol, double rtol, int maxIt, bool /*verbose*/)
{
    const int Kloc = nx_ * ny_;

    // 分配 / 重置 PCG 向量
    w_.assign(Kloc, 0.0);
    r_.assign(Kloc, 0.0);
    p_.assign(Kloc, 0.0);
    Ap_.assign(Kloc, 0.0);
    z_.assign(Kloc, 0.0);
    z_new_.assign(Kloc, 0.0);

    // 初始解 w0 = 0 => r0 = F
    r_ = F_;

    // z0 = D^{-1} r0
    solve_Dz_eq_r(r_, z_);
    p_ = z_;

    double zr_loc  = dotE_local(z_, r_);
    double zr_glob = 0.0;
    MPI_Allreduce(&zr_loc, &zr_glob,
                  1, MPI_DOUBLE, MPI_SUM, comm_);

    // 用初始残差的 E 范数做归一化
    double normF = normE_global(r_);
    if (normF < 1e-16) normF = 1.0;

    int it = 0;
    for (; it < maxIt; ++it) {
        // Ap = A p   （这里会根据是否定义 USE_CUDA 走 GPU 或 CPU 版本）:contentReference[oaicite:1]{index=1}
        apply_A(p_, Ap_);

        // 计算 p^T A p
        double pAp_loc  = dotE_local(p_, Ap_);
        double pAp_glob = 0.0;
        MPI_Allreduce(&pAp_loc, &pAp_glob,
                      1, MPI_DOUBLE, MPI_SUM, comm_);

        double alpha = zr_glob / pAp_glob;

        // w = w + alpha * p
        // r = r - alpha * Ap
        for (int k = 0; k < Kloc; ++k) {
            w_[k] += alpha * p_[k];
            r_[k] -= alpha * Ap_[k];
        }

        // 检查收敛：绝对残差 / 相对残差
        double norm_r = normE_global(r_);
        if (norm_r < tol || norm_r / normF < rtol) {
            ++it;          // 当前这一步也算一次迭代
            break;
        }

        // 预条件：z_new = D^{-1} r
        solve_Dz_eq_r(r_, z_new_);

        double zr_new_loc  = dotE_local(z_new_, r_);
        double zr_new_glob = 0.0;
        MPI_Allreduce(&zr_new_loc, &zr_new_glob,
                      1, MPI_DOUBLE, MPI_SUM, comm_);

        double beta = zr_new_glob / zr_glob;

        // p = z_new + beta * p
        for (int k = 0; k < Kloc; ++k) {
            p_[k] = z_new_[k] + beta * p_[k];
        }

        z_      = z_new_;
        zr_glob = zr_new_glob;
    }

    // 记录一下迭代次数，方便之后写到报告里:contentReference[oaicite:2]{index=2}
    stats_.iters = it;
    return it;
}
// =======================================
//          收集并保存结果
// =======================================
void MPISolver2D::gather_and_save(const std::string& fname) const
{
    // 当前进程本地有 nx_ * ny_ 个内部点
    int Kloc = nx_ * ny_;
    std::vector<double> loc(Kloc);

    // 把当前进程负责的内部解 w_，压成一个连续的一维数组 loc
    for (int jj = 0; jj < ny_; ++jj) {
        for (int ii = 0; ii < nx_; ++ii) {
            // id_halo(ii+1, jj+1) 是带 halo 的索引
            // id_interior(ii, jj) 是压平后的本地索引 [0, Kloc)
            loc[id_interior(ii, jj)] = w_[id_interior(ii, jj)];
        }
    }

    // 每个进程本地有多少个数据要发
    int locCount = Kloc;
    std::vector<int> counts(size_), displs(size_);

    // 把每个进程的 locCount 先收集到 0 号进程
    MPI_Gather(&locCount, 1, MPI_INT,
               counts.data(), 1, MPI_INT,
               0, comm_);

    // 0 号进程根据 counts 计算 Gatherv 的偏移量 displs
    if (rank_ == 0) {
        displs[0] = 0;
        for (int i = 1; i < size_; ++i) {
            displs[i] = displs[i - 1] + counts[i - 1];
        }
    }

    // 0 号进程准备好接收所有数据的数组
    std::vector<double> global;
    int totalCount = 0;
    if (rank_ == 0) {
        totalCount = displs[size_ - 1] + counts[size_ - 1];
        global.resize(totalCount);
    }

    // 把各进程的 loc 拼到 0 号进程的 global 里
    MPI_Gatherv(loc.data(), locCount, MPI_DOUBLE,
                global.data(), counts.data(), displs.data(), MPI_DOUBLE,
                0, comm_);

    // 0 号进程把 global 写到文件里
    if (rank_ == 0) {
        std::ofstream f(fname);
        for (double v : global) {
            f << v << "\n";
        }
        std::cerr << "Saved solution to " << fname << "\n";
    }
}


