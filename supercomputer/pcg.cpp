#include "solver.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <chrono>

Solver2D::Solver2D(int M, int N) : M_(M), N_(N) {
    h1_ = (B1_ - A1_) / M_;
    h2_ = (B2_ - A2_) / N_;
    h_  = std::max(h1_, h2_);
    eps_ = h_ * h_; // eps = h^2

    a_.assign((M_+1)*(N_+1), 0.0);
    b_.assign((M_+1)*(N_+1), 0.0);
    F_.assign((M_-1)*(N_-1), 0.0);
    Ddiag_.assign((M_-1)*(N_-1), 0.0);
}

void Solver2D::build_ab() {
    for (int i = 1; i <= M_; ++i) {
        double x_edge = A1_ + (i - 0.5) * h1_;     // x_{i-1/2}
        for (int j = 1; j <= N_; ++j) {
            // ---- a(i,j)：竖向半边 [y_{j-1/2},y_{j+1/2}] 上 k 的平均 ----
            double y_low = A2_ + (j - 0.5) * h2_;  // y_{j-1/2}
            double y_up  = y_low + h2_;            // y_{j+1/2}
            double aij = 0.0;
            if (x_edge <= 0.0) {
                aij = 1.0; // 竖段全在 D
            } else {
                // 仅 y<=0 的部分在 D
                double ell_y = std::max(0.0, std::min(y_up, 0.0) - y_low);
                aij = (ell_y / h2_) + (1.0 - ell_y / h2_) / eps_;
            }
            a_[idxAB(i,j)] = aij;

            // ---- b(i,j)：横向半边 [x_{i-1/2},x_{i+1/2}] 上 k 的平均 ----
            double y_edge = A2_ + (j - 0.5) * h2_; // y_{j-1/2}
            double x_low  = A1_ + (i - 0.5) * h1_; // x_{i-1/2}
            double x_up   = x_low + h1_;           // x_{i+1/2}
            double bij = 0.0;
            if (y_edge <= 0.0) {
                bij = 1.0; // 横段全在 D
            } else {
                // 仅 x<=0 的部分在 D
                double ell_x = std::max(0.0, std::min(x_up, 0.0) - x_low);
                bij = (ell_x / h1_) + (1.0 - ell_x / h1_) / eps_;
            }
            b_[idxAB(i,j)] = bij;
        }
    }
}

//单元落在 D 的面积占比
void Solver2D::build_F() {
    for (int i = 1; i <= M_-1; ++i) {
        double xL = A1_ + (i - 0.5) * h1_; // x_{i-1/2}
        double xR = xL + h1_;              // x_{i+1/2}
        double Lx_pos = std::max(0.0, xR) - std::max(0.0, xL); // 单元在 x>0 的长度
        for (int j = 1; j <= N_-1; ++j) {
            double yB = A2_ + (j - 0.5) * h2_; // y_{j-1/2}
            double yT = yB + h2_;              // y_{j+1/2}
            double Ly_pos = std::max(0.0, yT) - std::max(0.0, yB); // 单元在 y>0 的长度
            double fij = 1.0 - (Lx_pos * Ly_pos) / (h1_ * h2_);
            if (fij < 0.0) fij = 0.0;
            if (fij > 1.0) fij = 1.0;
            F_[idxU(i,j)] = fij;
        }
    }
}

void Solver2D::build_Ddiag() {
    for (int i = 1; i <= M_-1; ++i) {
        for (int j = 1; j <= N_-1; ++j) {
            double d = (a_[idxAB(i+1,j)] + a_[idxAB(i,j)]) / (h1_*h1_)
                     + (b_[idxAB(i,j+1)] + b_[idxAB(i,j)]) / (h2_*h2_);
            Ddiag_[idxU(i,j)] = d;
        }
    }
}

//守恒通量形式
void Solver2D::apply_A(const std::vector<double>& x, std::vector<double>& y) const {
    y.assign((M_-1)*(N_-1), 0.0);
    for (int i = 1; i <= M_-1; ++i) {
        for (int j = 1; j <= N_-1; ++j) {
            int id = idxU(i,j);
            double wC = x[id];
            double wL = (i==1     ) ? 0.0 : x[idxU(i-1,j)];
            double wR = (i==M_-1 ) ? 0.0 : x[idxU(i+1,j)];
            double wB = (j==1     ) ? 0.0 : x[idxU(i,j-1)];
            double wT = (j==N_-1 ) ? 0.0 : x[idxU(i,j+1)];

            double t_x =  a_[idxAB(i+1,j)] * (wR - wC) - a_[idxAB(i,j)] * (wC - wL);
            double t_y =  b_[idxAB(i,j+1)] * (wT - wC) - b_[idxAB(i,j)] * (wC - wB);
            y[id] = - t_x / (h1_*h1_) - t_y / (h2_*h2_);
        }
    }
}

double Solver2D::dotE(const std::vector<double>& u, const std::vector<double>& v) const {
    double s = 0.0;
    const size_t K = u.size();
    for (size_t k = 0; k < K; ++k) s += u[k]*v[k];
    return s * (h1_*h2_);
}
double Solver2D::normE(const std::vector<double>& u) const {
    return std::sqrt(dotE(u,u));
}

//  Dz=r
void Solver2D::solve_Dz_eq_r(const std::vector<double>& r, std::vector<double>& z) const {
    const size_t K = r.size();
    z.resize(K);
    for (size_t k = 0; k < K; ++k) z[k] = r[k] / Ddiag_[k];
}


int Solver2D::pcg(std::vector<double>& w, double tol, double rtol, int maxIt, bool verbose) {
    const int K = (M_-1)*(N_-1);
    if (maxIt < 0) maxIt = K;

    w.assign(K, 0.0);
    std::vector<double> r = F_;          // 初始残差 r = F - A*0 = F
    std::vector<double> z(K), p(K), Ap(K), z_new(K);

    // 预条件
    solve_Dz_eq_r(r, z);
    p = z;

    double zr = dotE(z, r);
    double normF = normE(F_);
    double H_prev = 0.0;

    for (int it = 0; it < maxIt; ++it) {
        apply_A(p, Ap);
        double pAp = dotE(p, Ap);
        if (pAp <= 0.0) {
            if (verbose) std::cerr << "[warn] p^T A p <= 0, restart to steepest descent.\n";
            p = z;
            apply_A(p, Ap);
            pAp = dotE(p, Ap);
        }

        double alpha = zr / pAp;

        // w_{k+1} = w_k + alpha p_k
        double step_norm = std::abs(alpha) * normE(p);
        for (int k = 0; k < K; ++k) w[k] += alpha * p[k];

        // r_{k+1} = r_k - alpha A p_k
        for (int k = 0; k < K; ++k) r[k] -= alpha * Ap[k];

        double nr = normE(r);
        if (verbose) {
            double H_curr = dotE(F_, w) + dotE(r, w);
            std::cout << "iter " << (it+1)
                      << " |Δw|_E=" << step_norm
                      << "  ||r||_E/||F||_E=" << (nr/(normF>0?normF:1.0))
                      << "  H=" << H_curr << "\n";
            if (H_curr + 1e-14*std::abs(H_prev) < H_prev) {
                std::cout << "  [restart] H(w) lost monotonicity. Reset direction.\n";
            }
            H_prev = H_curr;
        }
        if (step_norm < tol || nr <= rtol * (normF>0?normF:1.0)) {
            return it+1;
        }

        // z_{k+1} = D^{-1} r_{k+1}
        solve_Dz_eq_r(r, z_new);
        double zr_new = dotE(z_new, r);

        double beta = zr_new / zr;
        zr = zr_new;

        // p_{k+1} = z_{k+1} + beta p_k
        for (int k = 0; k < K; ++k) p[k] = z_new[k] + beta * p[k];
        z.swap(z_new);
    }
    return maxIt;
}


void Solver2D::save_csv(const std::string& fname, const std::vector<double>& w) const {
    std::ofstream ofs(fname);
    ofs << std::setprecision(16);
    for (int j = 0; j <= N_; ++j) {
        for (int i = 0; i <= M_; ++i) {
            double val = 0.0;
            if (isInterior(i,j)) val = w[idxU(i,j)]; // 边界=0
            ofs << val;
            if (i < M_) ofs << ",";
        }
        ofs << "\n";
    }
    ofs.close();
}
