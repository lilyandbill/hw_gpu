#include "domain_decomp.h"

#include <cmath>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <algorithm>

using std::vector;

// 选择 Px, Py：遍历 P 的因子，尽量满足：
// 1) 子块长宽比在 [1/2, 2] 之内；
// 2) 子块尽量接近正方形；
// 3) Px*Py = P。
void DomainDecomposer::choose_process_grid(int M, int N, int P,
                                           int& Px, int& Py)
{
    if (P <= 0) {
        throw std::invalid_argument("P must be > 0");
    }

    const int nxGlob = std::max(1, M - 1); // 全局内部点数
    const int nyGlob = std::max(1, N - 1);

    double bestCost = 1e300;
    int bestPx = 1, bestPy = P;

    // 枚举 P 的所有因子
    for (int px = 1; px <= P; ++px) {
        if (P % px != 0) continue;
        int py = P / px;

        // 估计在这种切分下，每块内部点数的范围
        int nx_floor = nxGlob / px;
        int nx_ceil  = (nxGlob + px - 1) / px; // ceil(nxGlob / px)
        int ny_floor = nyGlob / py;
        int ny_ceil  = (nyGlob + py - 1) / py; // ceil(nyGlob / py)

        // 取最“极端”的长宽比
        double r1 = (double)nx_ceil / std::max(1, ny_floor);
        double r2 = (double)nx_floor / std::max(1, ny_ceil);
        double r_max = std::max(r1, r2);
        double r_min = std::min(r1, r2);

        // 题目要求近似满足 1/2 <= nx/ny <= 2
        if (r_max > 2.0 || r_min < 0.5) {
            continue;
        }

        // 代价函数：越接近 1 越好；同时让 Px/Py 接近全局网格的长宽比
        double aspectGlob = (double)nxGlob / (double)nyGlob;
        double aspectGrid = (double)px / (double)py;
        double cost = (r_max - 1.0) * (r_max - 1.0)
                    + 0.01 * (aspectGrid - aspectGlob) * (aspectGrid - aspectGlob);

        if (cost < bestCost) {
            bestCost = cost;
            bestPx = px;
            bestPy = py;
        }
    }

    // 如果严格满足比例约束的组合一个都没有（某些极端小网格 + P 组合），
    // 则退而求其次：只保证 Px*Py=P 且 Px 接近 sqrt(P)
    if (bestCost == 1e300) {
        int pxBest = 1;
        double bestDiff = 1e300;
        for (int px = 1; px <= P; ++px) {
            if (P % px != 0) continue;
            double diff = std::abs((double)px - std::sqrt((double)P));
            if (diff < bestDiff) {
                bestDiff = diff;
                pxBest = px;
            }
        }
        bestPx = pxBest;
        bestPy = P / pxBest;
    }

    Px = bestPx;
    Py = bestPy;
}

// 把 [1..nGlob] 分成 P 段，使得每段长度相差不超过 1。
// cuts[k] 是第 k 段的起始位置（1-based），cuts[P] = nGlob+1。
void DomainDecomposer::split_1d(int nGlob, int P, vector<int>& cuts)
{
    if (P <= 0) throw std::invalid_argument("split_1d: P must be > 0");
    if (nGlob <= 0) nGlob = 1;

    cuts.resize(P + 1);

    int base = nGlob / P;
    int rem  = nGlob % P;

    int pos = 1;
    cuts[0] = pos;
    for (int k = 0; k < P; ++k) {
        int len = base + (k < rem ? 1 : 0);
        pos += len;
        cuts[k + 1] = pos; // 最后 cuts[P] = nGlob + 1
    }
}

// 构造函数：确定 Px,Py 并在两维上切分
DomainDecomposer::DomainDecomposer(int M, int N, int P)
    : M_(M), N_(N), P_(P), Px_(1), Py_(P)
{
    if (M_ < 2 || N_ < 2) {
        throw std::invalid_argument("DomainDecomposer: M and N must be >= 2");
    }

    // 选择 Px,Py
    choose_process_grid(M_, N_, P_, Px_, Py_);

    // 在 x,y 方向上对内部点 1..M-1, 1..N-1 做一维切分
    split_1d(M_ - 1, Px_, cutsX_);
    split_1d(N_ - 1, Py_, cutsY_);
}

// 通过二维坐标 (ix,iy) 取子域
Subdomain DomainDecomposer::subdomain_at(int ix, int iy) const
{
    if (ix < 0 || ix >= Px_ || iy < 0 || iy >= Py_) {
        throw std::out_of_range("subdomain_at: index out of range");
    }

    Subdomain s;
    s.px = ix;
    s.py = iy;
    s.rank = iy * Px_ + ix; // 按 y 方向慢、x 方向快的顺序编号

    s.i0 = cutsX_[ix];
    s.i1 = cutsX_[ix + 1] - 1;
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

// 打印分块汇总信息（用于调试和完成第 4 题实验部分）
void DomainDecomposer::print_summary(std::ostream& os) const
{
    os << "==== Domain Decomposition Summary ====\n";
    os << "Global interior grid: " << (M_ - 1) << " x " << (N_ - 1) << "\n";
    os << "Total subdomains: " << P_
       << "  Grid layout (Px x Py): " << Px_ << " x " << Py_ << "\n";

    os << "X cuts (" << Px_ << " segments): ";
    for (int k = 0; k < Px_; ++k) {
        os << cutsX_[k];
        if (k + 1 < Px_) os << " ";
    }
    os << " (last+1=" << cutsX_[Px_] << ")\n";

    os << "Y cuts (" << Py_ << " segments): ";
    for (int k = 0; k < Py_; ++k) {
        os << cutsY_[k];
        if (k + 1 < Py_) os << " ";
    }
    os << " (last+1=" << cutsY_[Py_] << ")\n\n";

    os << std::left
       << std::setw(6)  << "Rank"
       << std::setw(12) << "coords"
       << std::setw(16) << "i-range"
       << std::setw(16) << "j-range"
       << std::setw(16) << "nx x ny"
       << std::setw(16) << "nx/ny"
       << "\n";

    for (int r = 0; r < P_; ++r) {
        Subdomain s = subdomain_for_rank(r);
        double aspect = (double)s.nx() / (double)s.ny();

        os << std::setw(6) << r;

        char buf[32];
        std::snprintf(buf, sizeof(buf), "(%d,%d)", s.px, s.py);
        os << std::setw(12) << buf;

        char buf_i[32], buf_j[32], buf_n[32];
        std::snprintf(buf_i, sizeof(buf_i), "[%d,%d]", s.i0, s.i1);
        std::snprintf(buf_j, sizeof(buf_j), "[%d,%d]", s.j0, s.j1);
        std::snprintf(buf_n, sizeof(buf_n), "%d x %d", s.nx(), s.ny());

        os << std::setw(16) << buf_i;
        os << std::setw(16) << buf_j;
        os << std::setw(16) << buf_n;
        os << std::setw(16) << std::fixed << std::setprecision(3) << aspect;
        os << "\n";
    }

    os << "======================================\n";
}
