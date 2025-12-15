// #pragma once

// #include <vector>
// #include <string>
// #include <mpi.h>
// #include "domain_decomp_GPU.h"

// // 记录运行结果（迭代数、时间等）
// struct MPIRunStats {
//     int M = 0, N = 0, P = 0;
//     int iters = 0;
//     double t_build = 0.0;
//     double t_solve = 0.0;
//     double t_total = 0.0;
// };

// class MPISolver2D {
// public:
//     // M, N：全局网格节点数（含边界），与 DomainDecomposer 中一致
//     MPISolver2D(int M, int N, MPI_Comm comm, const DomainDecomposer& decomp);
//     ~MPISolver2D();

//     // 构造系数矩阵/右端项（这里只构造 F_ 和 Ddiag_）
//     void build_ab_F_D();

//     // 预条件共轭梯度
//     // tol / rtol：绝对/相对残差阈值
//     // maxIt：最大迭代步数
//     // verbose：是否在各 rank 打印迭代信息（目前未使用，可以以后加 log）
//     int pcg(double tol, double rtol, int maxIt, bool verbose = false);

//     // 收集各进程解并保存到文件
//     void gather_and_save(const std::string& fname) const;

//     const MPIRunStats& stats() const { return stats_; }

// private:
//     // 全局参数
//     int M_, N_;
//     MPI_Comm comm_;
//     int rank_ = 0, size_ = 1;
//     Subdomain sub_; // 该进程负责的子域

//     // 局部尺寸
//     int nx_ = 0, ny_ = 0;           // 内部点数量（不含边界）
//     int halo_nx_ = 0, halo_ny_ = 0; // 带幽灵层尺寸（nx_+2, ny_+2）

//     // 网格常量
//     double A1_ = -1.0, B1_ = 1.0;
//     double A2_ = -1.0, B2_ = 1.0;
//     double h1_ = 0.0, h2_ = 0.0;    // 网格步长
//     double h_  = 0.0;               // max(h1_, h2_)
//     double eps_ = 0.0;              // h_*h_

//     // 系数（目前只真正用到了 F_ 和 Ddiag_，aL_/aR_/bB_/bT_ 预留）
//     std::vector<double> aL_, aR_, bB_, bT_;
//     std::vector<double> F_, Ddiag_;

//     // PCG 向量（这里全部按 “内部点大小 = nx_*ny_” 存储）
//     std::vector<double> w_, r_, p_, Ap_, z_, z_new_;

//     // 邻居 rank
//     int nbr_left_   = MPI_PROC_NULL;
//     int nbr_right_  = MPI_PROC_NULL;
//     int nbr_bottom_ = MPI_PROC_NULL;
//     int nbr_top_    = MPI_PROC_NULL;

//     MPIRunStats stats_;

//     // 工具函数：内部/halo 索引
//     inline int id_interior(int ii, int jj) const { return jj * nx_ + ii; }
//     inline int id_halo(int ii, int jj) const { return jj * halo_nx_ + ii; }

//     void setup_neighbors(const DomainDecomposer& decomp);

//     // 对带 halo 的向量做四方向 halo 交换
//     // u_halo 长度必须是 halo_nx_ * halo_ny_
//     void exchange_halo(std::vector<double>& u_halo) const;

//     // 应用离散算子 A：Au = A * u（u, Au 都是内部尺寸 nx_*ny_）
//     // 内部会构造 halo + 调用 GPU kernel（如果定义了 USE_CUDA）
//     void apply_A(const std::vector<double>& u, std::vector<double>& Au);

//     // 局部内积（E 范数中的内积，不含 h1*h2）
//     double dotE_local(const std::vector<double>& u,
//                       const std::vector<double>& v) const;

//     // 全局 E 范数：||u||_E = sqrt( sum |u|^2 * h1*h2 )
//     double normE_global(const std::vector<double>& u) const;

//     // 预条件：解 D z = r，这里 Ddiag_ 是对角元
//     void solve_Dz_eq_r(const std::vector<double>& r,
//                        std::vector<double>& z) const;

//     // GPU 中间缓冲：带 halo 的 u 和内部 Au
//     double* d_u_halo_ = nullptr; // 长度 halo_nx_ * halo_ny_
//     double* d_Au_     = nullptr; // 长度 nx_ * ny_
//     bool gpu_buffers_allocated_ = false;

//     void allocate_gpu_buffers();
//     void free_gpu_buffers();
// };
#pragma once

#include <vector>
#include <string>
#include <mpi.h>
#include <cuda_runtime.h>

#include "domain_decomp_GPU.h"

// 用于记录统计信息（迭代次数等）
struct SolverStats {
    int M = 0;
    int N = 0;
    int P = 0;
    int iters = 0;
};

class MPISolver2D {
public:
    MPISolver2D(int M, int N, MPI_Comm comm,
                const DomainDecomposer& decomp);
    ~MPISolver2D();

    // 组装系数矩阵（对角预条件）、右端项等（你原来的函数）
    void build_ab_F_D();

    // PCG 求解
    int pcg(double tol, double rtol, int maxIt, bool verbose);

    // 收集并输出解到文件
    void gather_and_save(const std::string& fname) const;

    const SolverStats& stats() const { return stats_; }

private:
    // ====== 基本 MPI / 网格信息 ======
    int M_;          // 全局网格节点数（含边界）
    int N_;
    MPI_Comm comm_;
    int rank_ = 0;
    int size_ = 1;

    Subdomain sub_;  // 本进程子域
    int nx_ = 0;     // 本子域内部点数（x 方向）
    int ny_ = 0;     // 本子域内部点数（y 方向）

    int halo_nx_ = 0;  // = nx_ + 2
    int halo_ny_ = 0;  // = ny_ + 2

    // 物理参数 / 网格步长（与你原代码保持一致）
    double A1_ = 0.0, B1_ = 1.0;
    double A2_ = 0.0, B2_ = 1.0;
    double h1_ = 0.0, h2_ = 0.0, h_ = 0.0;
    double eps_ = 0.0;
    double inv_h2_ = 0.0; 
    // 邻居进程号（无邻居则 MPI_PROC_NULL）
    int nbr_left_   = MPI_PROC_NULL;
    int nbr_right_  = MPI_PROC_NULL;
    int nbr_bottom_ = MPI_PROC_NULL;
    int nbr_top_    = MPI_PROC_NULL;

    // ====== CPU 端向量 ======
    std::vector<double> F_;      // 右端项
    std::vector<double> Ddiag_;  // 对角预条件
    std::vector<double> w_;      // 解向量（本子域内部）

    // ====== GPU 端向量（device pointers） ======
    double* d_w_    = nullptr;
    double* d_r_    = nullptr;
    double* d_p_    = nullptr;
    double* d_Ap_   = nullptr;
    double* d_z_    = nullptr;
    double* d_znew_ = nullptr;
    double* d_F_    = nullptr;
    double* d_D_    = nullptr;

    // halo 区
    double* d_u_halo_ = nullptr;

    // halo 发送 / 接收缓冲（device）
    double* d_left_send_   = nullptr;
    double* d_right_send_  = nullptr;
    double* d_bottom_send_ = nullptr;
    double* d_top_send_    = nullptr;

    double* d_left_recv_   = nullptr;
    double* d_right_recv_  = nullptr;
    double* d_bottom_recv_ = nullptr;
    double* d_top_recv_    = nullptr;

    // halo 发送 / 接收缓冲（host）
    std::vector<double> host_left_send_;
    std::vector<double> host_right_send_;
    std::vector<double> host_bottom_send_;
    std::vector<double> host_top_send_;

    std::vector<double> host_left_recv_;
    std::vector<double> host_right_recv_;
    std::vector<double> host_bottom_recv_;
    std::vector<double> host_top_recv_;

    // ====== CUDA stream / 标志 ======
    cudaStream_t stream_;
    bool gpu_allocated_ = false;

    // 统计
    SolverStats stats_;

private:
    // 计算本子域邻居
    void setup_neighbors(const DomainDecomposer& decomp);

    // 1D 编号 (ii, jj) -> idx, 对应 nx_ x ny_ 内部网格
    inline int id_interior(int ii, int jj) const {
        return jj * nx_ + ii;
    }

    // GPU 内存管理
    void allocate_gpu_vectors();
    void free_gpu_vectors();

    // 在 GPU 上进行 halo 交换（与 .cu 中的 pack/unpack/build 配合）
    void exchange_halo_gpu(const double* d_u);

    // 在 GPU 上应用矩阵 A：给定 d_u，返回 d_Au
    void apply_A_GPU(const double* d_u, double* d_Au);
};
