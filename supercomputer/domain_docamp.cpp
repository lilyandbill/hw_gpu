// domain_decomp.cpp
#include "domain_decomp.h"
#include <cmath>
#include <algorithm>

void DomainDecomposer::choose_PxPy(int M, int N, int P, int &Px, int &Py) {
    double best_cost = 1e30;
    int best_px = 1, best_py = P;

    for (int px = 1; px <= P; ++px) {
        if (P % px) continue;
        int py = P / px;

        int mx_floor = (M-1) / px;
        int mx_ceil  = (M-1 + px - 1) / px;
        int ny_floor = (N-1) / py;
        int ny_ceil  = (N-1 + py - 1) / py;

        double r1 = (double)mx_ceil / std::max(1,ny_floor);
        double r2 = (double)mx_floor / std::max(1,ny_ceil);
        double r_max = std::max(r1,r2);
        double r_min = std::min(r1,r2);

        if (r_max > 2.0 || r_min < 0.5) continue;

        double cost = r_max - 1.0;
        if (cost < best_cost) {
            best_cost = cost;
            best_px = px; best_py = py;
        }
    }
    Px = best_px; Py = best_py;
}

// 把 [1..nGlob] 平均分给 P 段，差值 ≤ 1
void DomainDecomposer::split_1d(int nGlob, int P, std::vector<int>& cuts) {
    cuts.resize(P+1);
    int base = nGlob / P;
    int rem  = nGlob % P;
    int pos = 1;
    cuts[0] = 1;
    for (int p = 0; p < P; ++p) {
        int len = base + (p < rem ? 1 : 0);
        pos += len;
        cuts[p+1] = pos;
    }
}

DomainDecomposer::DomainDecomposer(int M, int N, MPI_Comm comm)
    : M_(M), N_(N), cart_(MPI_COMM_NULL) {

    MPI_Comm_rank(comm, &rank_);
    MPI_Comm_size(comm, &size_);

    choose_PxPy(M_, N_, size_, Px_, Py_);

    int dims[2] = {Px_, Py_};
    int periods[2] = {0,0};
    MPI_Cart_create(comm, 2, dims, periods, 1, &cart_);

    MPI_Comm_rank(cart_, &rank_);
    MPI_Cart_coords(cart_, rank_, 2, coords_);

    int px = coords_[0], py = coords_[1];

    // 按内部点数 M-1, N-1 分
    std::vector<int> cutsX, cutsY;
    split_1d(M_-1, Px_, cutsX); // cutsX.size() = Px+1
    split_1d(N_-1, Py_, cutsY);

    sub_.ix = px; sub_.iy = py;
    sub_.i0 = cutsX[px];
    sub_.i1 = cutsX[px+1] - 1;
    sub_.j0 = cutsY[py];
    sub_.j1 = cutsY[py+1] - 1;

    // 邻居
    int ncoords[2];
    MPI_Cart_shift(cart_, 0,  1, &nbr_left_,  &nbr_right_);
    MPI_Cart_shift(cart_, 1,  1, &nbr_bottom_,&nbr_top_);
}
