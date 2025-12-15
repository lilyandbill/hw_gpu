#pragma once
#include <vector>
#include <string>
#include <mpi.h>
#include "domain_decomp.h"

class MPISolver2D {
public:
    MPISolver2D(int M, int N, MPI_Comm comm, const DomainDecomposer& decomp);

    void build_ab_F_D();
    int  pcg(double tol, double rtol, int maxIt, bool verbose=false);
    void gather_and_save(const std::string& fname) const;

private:
    // 网格
    int M_, N_;
    int nx_, ny_, halo_nx_, halo_ny_;
    int rank_, size_;
    MPI_Comm comm_;
    Subdomain sub_;

    // 参数
    double A1_=-1, B1_=1, A2_=-1, B2_=1;
    double h1_, h2_, h_, eps_;

    // 邻居
    int nbr_left_, nbr_right_, nbr_bottom_, nbr_top_;

    // 数据
    std::vector<double> F_, Ddiag_;
    std::vector<double> w_, r_, p_, Ap_, z_, z_new_;

    // 工具
    inline int id_halo(int i,int j) const { return j*halo_nx_ + i; }
    inline int id_interior(int i,int j) const { return j*nx_ + i; }

    void setup_neighbors(const DomainDecomposer& decomp);
    void exchange_halo(std::vector<double>& u) const;
    void apply_A(const std::vector<double>& u, std::vector<double>& Au) const;
    double dotE_local(const std::vector<double>& u,const std::vector<double>& v) const;
    double normE_global(const std::vector<double>& u) const;
    void solve_Dz_eq_r(const std::vector<double>& r, std::vector<double>& z) const;
};
