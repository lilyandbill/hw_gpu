// pcg_openmp.cpp — 实现文件：Solver2D + PCG + OpenMP + 计时装饰
// 并行编译：g++ -O3 -std=c++17 -fopenmp pcg_openmp.cpp main_openmp.cpp -o pcg_openmp
// 运行建议：OMP_PROC_BIND=true OMP_PLACES=cores OMP_DYNAMIC=false ./pcg_openmp

#include "solver_openmp.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <chrono>
#include <string>
#include <vector>

#ifdef _OPENMP
  #include <omp.h>
#endif

// ====== 并行开关（可在编译时用 -D 覆盖） ======
#ifndef PAR_BUILD_AB
#define PAR_BUILD_AB 1
#endif
#ifndef PAR_BUILD_F
#define PAR_BUILD_F 1
#endif
#ifndef PAR_BUILD_D
#define PAR_BUILD_D 1
#endif
#ifndef PAR_APPLY_A
#define PAR_APPLY_A 1
#endif
#ifndef PAR_DOTE
#define PAR_DOTE 1
#endif
#ifndef PAR_DZ
#define PAR_DZ 1
#endif

// ===================== Solver2D 实现 =====================

Solver2D::Solver2D(int M, int N)
    : M_(M), N_(N),
      A1_(-1.0), B1_(1.0), A2_(-1.0), B2_(1.0) {
    h1_ = (B1_ - A1_) / M_;
    h2_ = (B2_ - A2_) / N_;
    h_  = std::max(h1_, h2_);
    eps_ = h_ * h_; // 虚域法参数 eps = h^2

    a_.assign((M_+1)*(N_+1), 0.0);
    b_.assign((M_+1)*(N_+1), 0.0);
    F_.assign((M_-1)*(N_-1), 0.0);
    Ddiag_.assign((M_-1)*(N_-1), 0.0);
}

// a,b：段平均（靴形域闭式）
void Solver2D::build_ab() {
#if defined(_OPENMP) && PAR_BUILD_AB
#pragma omp parallel for collapse(2) schedule(static) default(none) \
    shared(A1_,A2_,h1_,h2_,eps_,M_,N_,a_,b_)
#endif
    for (int i = 1; i <= M_; ++i) {
        for (int j = 1; j <= N_; ++j) {
            // ---- 竖边 a(i,j) ----
            const double x_edge = A1_ + (i - 0.5) * h1_; // x_{i-1/2}
            const double y_low  = A2_ + (j - 0.5) * h2_; // y_{j-1/2}
            const double y_up   = y_low + h2_;           // y_{j+1/2}

            double aij;
            if (x_edge <= 0.0) {
                aij = 1.0;
            } else {
                const double ell_y = std::max(0.0, std::min(y_up, 0.0) - y_low);
                aij = (ell_y / h2_) + (1.0 - ell_y / h2_) / eps_;
            }
            a_[idxAB(i,j)] = aij;

            // ---- 横边 b(i,j) ----
            const double y_edge = A2_ + (j - 0.5) * h2_; // y_{j-1/2}
            const double x_low  = A1_ + (i - 0.5) * h1_; // x_{i-1/2}
            const double x_up   = x_low + h1_;           // x_{i+1/2}

            double bij;
            if (y_edge <= 0.0) {
                bij = 1.0;
            } else {
                const double ell_x = std::max(0.0, std::min(x_up, 0.0) - x_low);
                bij = (ell_x / h1_) + (1.0 - ell_x / h1_) / eps_;
            }
            b_[idxAB(i,j)] = bij;
        }
    }
}

// F：单元在 D 的面积占比（靴形域闭式）
void Solver2D::build_F() {
#if defined(_OPENMP) && PAR_BUILD_F
#pragma omp parallel for collapse(2) schedule(static) default(none) \
    shared(A1_,A2_,h1_,h2_,M_,N_,F_)
#endif
    for (int i = 1; i <= M_-1; ++i) {
        for (int j = 1; j <= N_-1; ++j) {
            const double xL = A1_ + (i - 0.5) * h1_, xR = xL + h1_;
            const double yB = A2_ + (j - 0.5) * h2_, yT = yB + h2_;
            const double Lx_pos = std::max(0.0, xR) - std::max(0.0, xL);
            const double Ly_pos = std::max(0.0, yT) - std::max(0.0, yB);
            double fij = 1.0 - (Lx_pos * Ly_pos) / (h1_ * h2_);
            if (fij < 0.0) fij = 0.0;
            if (fij > 1.0) fij = 1.0;
            F_[idxU(i,j)] = fij;
        }
    }
}

// D 对角（Jacobi 预条件）
void Solver2D::build_Ddiag() {
#if defined(_OPENMP) && PAR_BUILD_D
#pragma omp parallel for collapse(2) schedule(static) default(none) \
    shared(M_,N_,a_,b_,h1_,h2_,Ddiag_)
#endif
    for (int i = 1; i <= M_-1; ++i) {
        for (int j = 1; j <= N_-1; ++j) {
            const double d = (a_[idxAB(i+1,j)] + a_[idxAB(i,j)]) / (h1_*h1_)
                           + (b_[idxAB(i,j+1)] + b_[idxAB(i,j)]) / (h2_*h2_);
            Ddiag_[idxU(i,j)] = d;
        }
    }
}

// y = A x（矩阵-自由，保守型五点）
void Solver2D::apply_A(const std::vector<double>& x, std::vector<double>& y) const {
    y.assign((M_-1)*(N_-1), 0.0); // 并行外清零，避免竞争

#if defined(_OPENMP) && PAR_APPLY_A
#pragma omp parallel for collapse(2) schedule(static) default(none) \
    shared(M_,N_,a_,b_,h1_,h2_,x,y)
#endif
    for (int i = 1; i <= M_-1; ++i) {
        for (int j = 1; j <= N_-1; ++j) {
            const int id = idxU(i, j);

            const double wC = x[id];
            const double wL = (i==1    ) ? 0.0 : x[idxU(i-1,j)];
            const double wR = (i==M_-1 ) ? 0.0 : x[idxU(i+1,j)];
            const double wB = (j==1    ) ? 0.0 : x[idxU(i,  j-1)];
            const double wT = (j==N_-1 ) ? 0.0 : x[idxU(i,  j+1)];

            const double t_x =  a_[idxAB(i+1,j)] * (wR - wC) - a_[idxAB(i,j)] * (wC - wL);
            const double t_y =  b_[idxAB(i,j+1)] * (wT - wC) - b_[idxAB(i,j)] * (wC - wB);

            y[id] = - t_x / (h1_*h1_) - t_y / (h2_*h2_);
        }
    }
}

// 带权内积与范数（E-范）
double Solver2D::dotE(const std::vector<double>& u, const std::vector<double>& v) const {
    double s = 0.0;
#if defined(_OPENMP) && PAR_DOTE
#pragma omp parallel for reduction(+:s) schedule(static) default(none) shared(u,v,h1_,h2_)
#endif
    for (int k = 0; k < (int)u.size(); ++k) s += u[k]*v[k];
    return s * (h1_*h2_);
}
double Solver2D::normE(const std::vector<double>& u) const { return std::sqrt(dotE(u,u)); }

// 预条件解 Dz = r（逐点相除）
void Solver2D::solve_Dz_eq_r(const std::vector<double>& r, std::vector<double>& z) const {
    z.resize(r.size());
#if defined(_OPENMP) && PAR_DZ
#pragma omp parallel for schedule(static) default(none) shared(r,z,Ddiag_)
#endif
    for (int k = 0; k < (int)r.size(); ++k) z[k] = r[k] / Ddiag_[k];
}

// PCG 主求解
// int Solver2D::pcg(std::vector<double>& w, double tol, double rtol, int maxIt, bool verbose) {
//     const int K = (M_-1)*(N_-1);
//     if (maxIt < 0) maxIt = K;

//     w.assign(K, 0.0);
//     std::vector<double> r = F_, z(K), p(K), Ap(K), z_new(K);

//     solve_Dz_eq_r(r, z);
//     p = z;

//     double zr = dotE(z, r);
//     const double normF = normE(F_);
//     double H_prev = 0.0;

//     for (int it = 0; it < maxIt; ++it) {
//         apply_A(p, Ap);
//         double pAp = dotE(p, Ap);
//         if (pAp <= 1e-300) { // 数值防护
//             if (verbose) std::cerr << "[warn] p^T A p <= 0, restart\n";
//             p = z;
//             apply_A(p, Ap);
//             pAp = dotE(p, Ap);
//             if (pAp <= 1e-300) break;
//         }

//         const double alpha = zr / pAp;

//         // w_{k+1} = w_k + alpha p_k
//         const double step_norm = std::abs(alpha) * normE(p);
//         for (int k = 0; k < K; ++k) w[k] += alpha * p[k];

//         // r_{k+1} = r_k - alpha A p_k
//         for (int k = 0; k < K; ++k) r[k] -= alpha * Ap[k];

//         const double nr = normE(r);
//         if (verbose) {
//             const double H_curr = dotE(F_, w) + dotE(r, w);
//             std::cout << "iter " << (it+1)
//                       << "  |Δw|_E=" << step_norm
//                       << "  ||r||_E/||F||_E=" << (nr/(normF>0?normF:1.0))
//                       << "  H=" << H_curr << "\n";
//             if (H_curr + 1e-14*std::abs(H_prev) < H_prev) {
//                 std::cout << "  [restart] H(w) lost monotonicity.\n";
//             }
//             H_prev = H_curr;
//         }
//         if (step_norm < tol || nr <= rtol * (normF>0?normF:1.0)) {
//             return it+1;
//         }

//         // 预条件
//         solve_Dz_eq_r(r, z_new);
//         const double zr_new = dotE(z_new, r);

//         const double beta = zr_new / zr;
//         zr = zr_new;

//         for (int k = 0; k < K; ++k) p[k] = z_new[k] + beta * p[k];
//         z.swap(z_new);
//     }
//     return maxIt;
// }
int Solver2D::pcg(std::vector<double>& w, double tol, double rtol, int maxIt, bool verbose) {
    const int K = (M_-1)*(N_-1);
    if (maxIt < 0) maxIt = K;

    w.assign(K, 0.0);

#ifndef _OPENMP
    // ===== 串行路径（保持与你原来一致） =====
    std::vector<double> r = F_, z(K), p(K), Ap(K), z_new(K);

    solve_Dz_eq_r(r, z);
    p = z;

    double zr = dotE(z, r);
    const double normF = normE(F_);
    for (int it = 0; it < maxIt; ++it) {
        apply_A(p, Ap);
        const double pAp = dotE(p, Ap);
        if (pAp <= 1e-300) return it;

        const double alpha = zr / pAp;

        // 步长准则 (|Δw|_E = |alpha| * ||p||_E)
        const double p2   = dotE(p, p);
        const double step = std::abs(alpha) * std::sqrt(p2);
        if (step < tol) return it+1;

        // w += αp；r -= αAp
        for (int k = 0; k < K; ++k) { w[k] += alpha * p[k]; r[k] -= alpha * Ap[k]; }

        // 相对残差
        const double nr = normE(r);
        if (nr <= rtol * (normF>0?normF:1.0)) return it+1;

        // 预条件与更新方向
        solve_Dz_eq_r(r, z_new);
        const double zr_new = dotE(z_new, r);
        const double beta   = zr_new / zr;
        zr = zr_new;
        for (int k = 0; k < K; ++k) p[k] = z_new[k] + beta * p[k];
        z.swap(z_new);
    }
    return maxIt;

#else
    // ===== 并行路径：持久线程团队 + 循环融合 =====
    std::vector<double> r = F_, z(K), p(K), Ap(K), z_new(K);

    double zr = 0.0, normF = 0.0;
    double alpha_shared = 0.0, beta_shared = 0.0;   // ☆ 作为 shared 变量供所有线程读取
    int    iters_out = maxIt;
    bool   stop_flag = false;

    // 这些是每轮用到的 reduction 变量（放并行区外，作为 shared）
    double zr_local = 0.0, nF2_local = 0.0;
    double pAp_local = 0.0, p2_local = 0.0;
    double nr2_local = 0.0, zr_new_local = 0.0;

    #pragma omp parallel shared(zr, normF, alpha_shared, beta_shared, iters_out, stop_flag, \
                                zr_local, nF2_local, pAp_local, p2_local, nr2_local, zr_new_local, \
                                w, r, z, p, Ap, z_new)
    {
        // z = D^{-1} r
        #pragma omp for schedule(static)
        for (int k = 0; k < K; ++k) z[k] = r[k] / Ddiag_[k];

        // 初始 zr、||F||
        #pragma omp single
        { zr_local = 0.0; nF2_local = 0.0; }
        #pragma omp for reduction(+:zr_local,nF2_local) schedule(static)
        for (int k = 0; k < K; ++k) {
            zr_local  += z[k]*r[k];
            nF2_local += F_[k]*F_[k];
        }
        #pragma omp single
        {
            const double wgt = (h1_*h2_);
            zr = zr_local * wgt;
            normF = std::sqrt(nF2_local * wgt);
            p = z;
        }

        for (int it = 0; it < maxIt; ++it) {
            if (stop_flag) break;

            // ---- Ap = A p（双层循环并行）----
            #pragma omp for collapse(2) schedule(static)
            for (int j = 1; j <= N_-1; ++j) {
                for (int i = 1; i <= M_-1; ++i) {
                    const int id = idxU(i,j);
                    const double wC = p[id];
                    const double wL = (i==1    ) ? 0.0 : p[idxU(i-1,j)];
                    const double wR = (i==M_-1 ) ? 0.0 : p[idxU(i+1,j)];
                    const double wB = (j==1    ) ? 0.0 : p[idxU(i,  j-1)];
                    const double wT = (j==N_-1 ) ? 0.0 : p[idxU(i,  j+1)];
                    const double t_x =  a_[idxAB(i+1,j)]*(wR - wC) - a_[idxAB(i,j)]*(wC - wL);
                    const double t_y =  b_[idxAB(i,  j+1)]*(wT - wC) - b_[idxAB(i,j)]*(wC - wB);
                    Ap[id] = - t_x/(h1_*h1_) - t_y/(h2_*h2_);
                }
            }

            // ---- 同一趟遍历做 pAp 和 ||p||^2（两者都需要）----
            #pragma omp single
            { pAp_local = 0.0; p2_local = 0.0; }
            #pragma omp for reduction(+:pAp_local,p2_local) schedule(static)
            for (int k = 0; k < K; ++k) {
                pAp_local += p[k]*Ap[k];
                p2_local  += p[k]*p[k];
            }

            // 计算 alpha 与步长准则
            #pragma omp single
            {
                const double wgt = (h1_*h2_);
                const double pAp = pAp_local * wgt;
                if (pAp <= 1e-300) { stop_flag = true; iters_out = it; }
                else {
                    alpha_shared = zr / pAp;
                    const double step = std::abs(alpha_shared) * std::sqrt(p2_local * wgt);
                    if (step < tol) { stop_flag = true; iters_out = it+1; }
                }
            }
            #pragma omp barrier
            if (stop_flag) break;

            // ---- 融合：w += αp；r -= αAp；累计 ||r||² ----
            #pragma omp single
            { nr2_local = 0.0; }
            #pragma omp for reduction(+:nr2_local) schedule(static)
            for (int k = 0; k < K; ++k) {
                w[k] += alpha_shared * p[k];
                r[k] -= alpha_shared * Ap[k];
                nr2_local += r[k]*r[k];
            }

            // 相对残差准则
            #pragma omp single
            {
                const double nr = std::sqrt(nr2_local * (h1_*h2_));
                if (nr <= rtol * (normF>0?normF:1.0)) { stop_flag = true; iters_out = it+1; }
            }
            #pragma omp barrier
            if (stop_flag) break;

            // ---- 融合：z_new = D^{-1} r；累计 (z_new,r) ----
            #pragma omp single
            { zr_new_local = 0.0; }
            #pragma omp for reduction(+:zr_new_local) schedule(static)
            for (int k = 0; k < K; ++k) {
                z_new[k] = r[k] / Ddiag_[k];
                zr_new_local += z_new[k]*r[k];
            }

            // 计算 beta 并更新方向
            #pragma omp single
            {
                const double zr_new = zr_new_local * (h1_*h2_);
                beta_shared = zr_new / zr;
                zr = zr_new;
            }
            #pragma omp for schedule(static)
            for (int k = 0; k < K; ++k) p[k] = z_new[k] + beta_shared * p[k];

            #pragma omp single
            { z.swap(z_new); }

            #pragma omp barrier
        } // for it
    } // parallel

    return iters_out;
#endif
}




// 保存整网格 CSV（边界=0）
void Solver2D::save_csv(const std::string& fname, const std::vector<double>& w) const {
    std::ofstream ofs(fname);
    ofs << std::setprecision(16);
    for (int j = 0; j <= N_; ++j) {
        for (int i = 0; i <= M_; ++i) {
            double val = 0.0;
            if (isInterior(i,j)) val = w[idxU(i,j)];
            ofs << val;
            if (i < M_) ofs << ",";
        }
        ofs << "\n";
    }
}

// ===================== TimedSolverDecorator 实现 =====================

static inline double now_seconds() {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    using clk = std::chrono::high_resolution_clock;
    return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
#endif
}

RunStats TimedSolverDecorator::run(int M, int N, int threads, double tol, double rtol,
                                   bool saveCsv, const std::string& csvNameBase) const {
#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif
    RunStats s; s.M=M; s.N=N; s.threads=threads; s.tol=tol; s.rtol=rtol;

    const double t0 = now_seconds();

    Solver2D solver(M, N);

    const double tb0 = now_seconds();
    solver.build_ab();
    solver.build_F();
    solver.build_Ddiag();
    const double tb1 = now_seconds();
    s.t_build = tb1 - tb0;

    std::vector<double> w;
    const double ts0 = now_seconds();
    s.iters = solver.pcg(w, tol, rtol, /*maxIt=*/-1, /*verbose=*/false);
    const double ts1 = now_seconds();
    s.t_solve = ts1 - ts0;

    if (saveCsv) {
        s.saved_csv = csvNameBase + "_" + std::to_string(M) + "x" + std::to_string(N)
                    + "_T" + std::to_string(threads) + ".csv";
        solver.save_csv(s.saved_csv, w);
    }

    const double t1 = now_seconds();
    s.t_total = t1 - t0;
    return s;
}
