#pragma once
#include <vector>
#include <string>
#include <mpi.h>
#include "domain_decomp.h"

// 记录运行结果（迭代数、时间等）
struct MPIRunStats {
    int M=0, N=0, P=0;
    int iters=0;
    double t_build=0.0;
    double t_solve=0.0;
    double t_total=0.0;
};

// =======================================
//         MPI 版 Poisson 方程求解器
// =======================================
class MPISolver2D {
public:
    MPISolver2D(int M, int N, MPI_Comm comm, const DomainDecomposer& decomp);

    void build_ab_F_D();  // 构造系数矩阵和右端项
    int pcg(double tol, double rtol, int maxIt, bool verbose=false);
    void gather_and_save(const std::string& fname) const;

    const MPIRunStats& stats() const { return stats_; }

private:
    // 全局参数
    int M_, N_;
    MPI_Comm comm_;
    int rank_, size_;
    Subdomain sub_;

    // 局部尺寸
    int nx_, ny_;           // 内部点
    int halo_nx_, halo_ny_; // 含幽灵层 = nx_+2, ny_+2

    // 网格常量
    double A1_=-1.0, B1_=1.0, A2_=-1.0, B2_=1.0;
    double h1_=0.0, h2_=0.0, h_=0.0, eps_=0.0;

    // 系数
    std::vector<double> aL_, aR_, bB_, bT_; // 左右上下边系数
    std::vector<double> F_, Ddiag_;         // 源项与预条件对角元

    // PCG 向量（含 halo）
    std::vector<double> w_, r_, p_, Ap_, z_, z_new_;

    // 邻居 rank
    int nbr_left_=MPI_PROC_NULL, nbr_right_=MPI_PROC_NULL;
    int nbr_bottom_=MPI_PROC_NULL, nbr_top_=MPI_PROC_NULL;

    MPIRunStats stats_;

    // 工具函数
    inline int id_interior(int ii,int jj) const { return jj*nx_ + ii; }
    inline int id_halo(int ii,int jj) const { return jj*halo_nx_ + ii; }

    void setup_neighbors(const DomainDecomposer& decomp);
    void exchange_halo(std::vector<double>& u) const;
    void apply_A(const std::vector<double>& u, std::vector<double>& Au) const;
    double dotE_local(const std::vector<double>& u, const std::vector<double>& v) const;
    double normE_global(const std::vector<double>& u) const;
    void solve_Dz_eq_r(const std::vector<double>& r, std::vector<double>& z) const;
};
