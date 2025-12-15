#include "domain_decomp_GPU.h"

#include <cmath>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <algorithm>
#include <cstdio>   // std::snprintf

using std::vector;

// ======================== 工具函数实现 ========================

// 在 (M-1)x(N-1) 的内部点上选择 Px,Py，使 Px*Py=P，块尽量“正方”
void DomainDecomposer::choose_process_grid(int M, int N, int P,
                                           int& Px, int& Py)
{
    if (P <= 0) {
        throw std::invalid_argument("choose_process_grid: P must be positive");
    }

    const int nX = std::max(1, M - 1); // 内部点个数
    const int nY = std::max(1, N - 1);

    double bestScore = 1e100;
    int bestPx = 1, bestPy = P;
    bool found = false;

    // 枚举 P 的因子
    for (int px = 1; px <= P; ++px) {
        if (P % px != 0) continue;
        int py = P / px;

        double subNx = double(nX) / px;
        double subNy = double(nY) / py;
        if (subNy <= 0.0) continue;

        double aspect = subNx / subNy;
        if (aspect < 1.0) aspect = 1.0 / aspect;  // >= 1

        // 优先约束在 [1/2, 2] 内（aspect <= 2）
        if (aspect > 2.0) continue;

        // 用子块长宽差作为评分，越小越好
        double score = std::fabs(subNx - subNy);
        if (score < bestScore) {
            bestScore = score;
            bestPx = px;
            bestPy = py;
            found = true;
        }
    }

    // 如果所有候选都不满足 aspect 限制，则退化成简单 sqrt 选择
    if (!found) {
        int px = 1;
        for (int k = 1; k <= P; ++k) {
            if (P % k == 0) {
                px = k;
                int py = P / k;
                // 挑一个接近 sqrt(P * nX / nY) 的
                // 这里用简单的“差值”来衡量
                double subNx = double(nX) / px;
                double subNy = double(nY) / py;
                double score = std::fabs(subNx - subNy);
                if (score < bestScore) {
                    bestScore = score;
                    bestPx = px;
                    bestPy = py;
                }
            }
        }
    }

    Px = bestPx;
    Py = bestPy;
}

// 把 [1..nGlob] 分成 P 段，长度差 <= 1，输出 cuts[0..P]，cuts[0]=1, cuts[P]=nGlob+1
void DomainDecomposer::split_1d(int nGlob, int P, vector<int>& cuts)
{
    if (nGlob <= 0 || P <= 0) {
        throw std::invalid_argument("split_1d: nGlob and P must be positive");
    }
    cuts.assign(P + 1, 0);

    int base = nGlob / P;
    int rem  = nGlob % P;

    int cur = 1;
    cuts[0] = cur;
    for (int p = 0; p < P; ++p) {
        int len = base + (p < rem ? 1 : 0);
        if (p + 1 <= P) {
            cur += len;
            cuts[p + 1] = cur;
        }
    }
    // 理论上 cuts[P] 应该等于 nGlob + 1
    cuts[P] = nGlob + 1;
}

// ======================== 构造函数 ========================

DomainDecomposer::DomainDecomposer(int M, int N, int P)
    : M_(M), N_(N), P_(P)
{
    if (M_ < 2 || N_ < 2) {
        throw std::invalid_argument("DomainDecomposer: M,N must be >= 2");
    }
    if (P_ <= 0) {
        throw std::invalid_argument("DomainDecomposer: P must be positive");
    }

    // 选 Px, Py
    choose_process_grid(M_, N_, P_, Px_, Py_);
    if (Px_ * Py_ != P_) {
        throw std::runtime_error("DomainDecomposer: Px*Py != P");
    }

    int nX = M_ - 1; // 内部点数量：i=1..M-1
    int nY = N_ - 1; // j=1..N-1
    split_1d(nX, Px_, cutsX_);
    split_1d(nY, Py_, cutsY_);
}

// ======================== 子域获取 ========================

Subdomain DomainDecomposer::subdomain_at(int ix, int iy) const
{
    if (ix < 0 || ix >= Px_ || iy < 0 || iy >= Py_) {
        throw std::out_of_range("subdomain_at: index out of range");
    }
    Subdomain s;
    s.px = ix;
    s.py = iy;
    s.rank = iy * Px_ + ix;

    s.i0 = cutsX_[ix];
    s.i1 = cutsX_[ix + 1] - 1; // [cutsX[ix], cutsX[ix+1]-1]
    s.j0 = cutsY_[iy];
    s.j1 = cutsY_[iy + 1] - 1;

    return s;
}

// 通过 rank 取子域
Subdomain DomainDecomposer::subdomain_for_rank(int rank) const
{
    if (rank < 0 || rank >= P_) {
        throw std::out_of_range("subdomain_for_rank: rank out of range");
    }
    int ix = rank % Px_;
    int iy = rank / Px_;
    return subdomain_at(ix, iy);
}

// ======================== 打印分块信息 ========================

void DomainDecomposer::print_summary(std::ostream& os) const
{
    os << "======================================\n";
    os << " Domain decomposition summary\n";
    os << "  Global internal grid: (M-1)x(N-1) = "
       << (M_ - 1) << " x " << (N_ - 1) << "\n";
    os << "  Processes: P = " << P_
       << "  (Px x Py = " << Px_ << " x " << Py_ << ")\n";
    os << "--------------------------------------\n";
    os << " rank  (px,py)       i-range         j-range"
          "         nx x ny         aspect\n";
    os << "--------------------------------------\n";

    for (int r = 0; r < P_; ++r) {
        Subdomain s = subdomain_for_rank(r);
        double nx_loc = s.nx();
        double ny_loc = s.ny();
        double aspect = (ny_loc > 0.0 ? nx_loc / ny_loc : 0.0);
        if (aspect < 0.0) aspect = -aspect;

        char buf_i[32], buf_j[32], buf_n[32];
        std::snprintf(buf_i, sizeof(buf_i), "[%d,%d]", s.i0, s.i1);
        std::snprintf(buf_j, sizeof(buf_j), "[%d,%d]", s.j0, s.j1);
        std::snprintf(buf_n, sizeof(buf_n), "%d x %d", s.nx(), s.ny());

        os << std::setw(4) << r << "  ";
        os << "(" << s.px << "," << s.py << ") ";

        os << std::setw(10) << buf_i;
        os << std::setw(14) << buf_j;
        os << std::setw(14) << buf_n;
        os << std::setw(14) << std::fixed << std::setprecision(3) << aspect;
        os << "\n";
    }

    os << "======================================\n";
}
