#pragma once

#include <vector>
#include <string>
#include <mpi.h>
#include "domain_decomp_GPU.h"

// 记录运行结果（迭代数、时间等）
struct MPIRunStats {
    int M = 0, N = 0, P = 0;
    int iters = 0;
    double t_build = 0.0;
    double t_solve = 0.0;
    double t_total = 0.0;
};

class MPISolver2D {
public:
    // M, N：全局网格节点数（含边界），与 DomainDecomposer 中一致
    MPISolver2D(int M, int N, MPI_Comm comm, const DomainDecomposer& decomp);
    ~MPISolver2D();

    // 构造系数矩阵/右端项（这里只构造 F_ 和 Ddiag_）
    void build_ab_F_D();

    // 预条件共轭梯度
    // tol / rtol：绝对/相对残差阈值
    // maxIt：最大迭代步数
    // verbose：是否在各 rank 打印迭代信息（目前未使用，可以以后加 log）
    int pcg(double tol, double rtol, int maxIt, bool verbose = false);

    // 收集各进程解并保存到文件
    void gather_and_save(const std::string& fname) const;

    const MPIRunStats& stats() const { return stats_; }

private:
    // 全局参数
    int M_, N_;
    MPI_Comm comm_;
    int rank_ = 0, size_ = 1;
    Subdomain sub_; // 该进程负责的子域

    // 局部尺寸
    int nx_ = 0, ny_ = 0;           // 内部点数量（不含边界）
    int halo_nx_ = 0, halo_ny_ = 0; // 带幽灵层尺寸（nx_+2, ny_+2）

    // 网格常量
    double A1_ = -1.0, B1_ = 1.0;
    double A2_ = -1.0, B2_ = 1.0;
    double h1_ = 0.0, h2_ = 0.0;    // 网格步长
    double h_  = 0.0;               // max(h1_, h2_)
    double eps_ = 0.0;              // h_*h_

    // 系数（目前只真正用到了 F_ 和 Ddiag_，aL_/aR_/bB_/bT_ 预留）
    std::vector<double> aL_, aR_, bB_, bT_;
    std::vector<double> F_, Ddiag_;

    // PCG 向量（这里全部按 “内部点大小 = nx_*ny_” 存储）
    std::vector<double> w_, r_, p_, Ap_, z_, z_new_;

    // 邻居 rank
    int nbr_left_   = MPI_PROC_NULL;
    int nbr_right_  = MPI_PROC_NULL;
    int nbr_bottom_ = MPI_PROC_NULL;
    int nbr_top_    = MPI_PROC_NULL;

    MPIRunStats stats_;

    // 工具函数：内部/halo 索引
    inline int id_interior(int ii, int jj) const { return jj * nx_ + ii; }
    inline int id_halo(int ii, int jj) const { return jj * halo_nx_ + ii; }

    void setup_neighbors(const DomainDecomposer& decomp);

    // 对带 halo 的向量做四方向 halo 交换
    // u_halo 长度必须是 halo_nx_ * halo_ny_
    void exchange_halo(std::vector<double>& u_halo) const;

    // 应用离散算子 A：Au = A * u（u, Au 都是内部尺寸 nx_*ny_）
    // 内部会构造 halo + 调用 GPU kernel（如果定义了 USE_CUDA）
    void apply_A(const std::vector<double>& u, std::vector<double>& Au);

    // 局部内积（E 范数中的内积，不含 h1*h2）
    double dotE_local(const std::vector<double>& u,
                      const std::vector<double>& v) const;

    // 全局 E 范数：||u||_E = sqrt( sum |u|^2 * h1*h2 )
    double normE_global(const std::vector<double>& u) const;

    // 预条件：解 D z = r，这里 Ddiag_ 是对角元
    void solve_Dz_eq_r(const std::vector<double>& r,
                       std::vector<double>& z) const;

    // GPU 中间缓冲：带 halo 的 u 和内部 Au
    double* d_u_halo_ = nullptr; // 长度 halo_nx_ * halo_ny_
    double* d_Au_     = nullptr; // 长度 nx_ * ny_
    bool gpu_buffers_allocated_ = false;

    void allocate_gpu_buffers();
    void free_gpu_buffers();
};
