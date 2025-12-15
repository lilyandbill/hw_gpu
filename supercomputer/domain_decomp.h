#pragma once

#include <vector>
#include <iosfwd>

// 描述一个子域的信息：对应全局内部点 i=1..M-1, j=1..N-1 中的一个矩形块
struct Subdomain {
    int rank = 0;   // 子域编号（0..P-1），将来可对应 MPI rank
    int px = 0;     // 在 Px x Py 网格中的 x 方向坐标（0..Px-1）
    int py = 0;     // 在 Px x Py 网格中的 y 方向坐标（0..Py-1）

    // 全局内部索引范围（包含端点），与 Solver2D 中的 i,j 一致：
    // i = 1..M-1, j = 1..N-1
    int i0 = 1;     // 全局内部 i 起点
    int i1 = 1;     // 全局内部 i 终点
    int j0 = 1;     // 全局内部 j 起点
    int j1 = 1;     // 全局内部 j 终点

    // 该子域内部未知点数
    int nx() const { return i1 - i0 + 1; } // x 方向
    int ny() const { return j1 - j0 + 1; } // y 方向
};

// 负责把 (M-1)x(N-1) 个内部点切成 P 个矩形子域
class DomainDecomposer {
public:
    // M, N 的含义与求解器中的一致：
    // 网格节点 i=0..M, j=0..N，内部未知数 i=1..M-1, j=1..N-1
    DomainDecomposer(int M, int N, int P);

    int M() const { return M_; }
    int N() const { return N_; }
    int P() const { return P_; }
    int Px() const { return Px_; }
    int Py() const { return Py_; }

    // 通过一维 rank (0<=rank<P) 获取子域
    Subdomain subdomain_for_rank(int rank) const;

    // 通过二维坐标 (ix,iy) 获取子域（0<=ix<Px, 0<=iy<Py）
    Subdomain subdomain_at(int ix, int iy) const;

    // 打印分块汇总信息，便于调试和验证
    void print_summary(std::ostream& os) const;

private:
    int M_;     // 全局网格节点数（含边界）x 方向
    int N_;     // 全局网格节点数（含边界）y 方向
    int P_;     // 子域总数（将来可等于 MPI 进程数）

    int Px_;    // x 方向子域数
    int Py_;    // y 方向子域数

    // 一维切分的端点：cutsX_[k] 是第 k 段的起始 i（内部索引，1-based）
    // 子域 ix 的 i 范围为 [cutsX_[ix], cutsX_[ix+1]-1]
    std::vector<int> cutsX_;
    std::vector<int> cutsY_;

    // 在 (M-1)x(N-1) 的内部点上选择 Px,Py，使 Px*Py=P，块尽量“正方”
    static void choose_process_grid(int M, int N, int P, int& Px, int& Py);

    // 把 [1..nGlob] 分成 P 段，长度差 <= 1，输出 cuts[0..P]，其中 cuts[0]=1, cuts[P]=nGlob+1
    static void split_1d(int nGlob, int P, std::vector<int>& cuts);
};
