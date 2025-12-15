#pragma once

#include <vector>
#include <iosfwd>

// 描述一个子域的信息：对应全局内部点 i=1..M-1, j=1..N-1 中的一个矩形块
struct Subdomain {
    int rank = 0;   // 子域编号（0..P-1），对应 MPI rank
    int px   = 0;   // 在 Px x Py 网格中的 x 方向坐标（0..Px-1）
    int py   = 0;   // 在 Px x Py 网格中的 y 方向坐标（0..Py-1）

    // 全局“内部”索引范围（包含端点），与 Solver 中的 i,j 一致：
    // i = 1..M-1, j = 1..N-1
    int i0 = 1;     // 全局内部 i 起点
    int i1 = 1;     // 全局内部 i 终点
    int j0 = 1;     // 全局内部 j 起点
    int j1 = 1;     // 全局内部 j 终点

    // 本子域内部点数量（不含边界）
    int nx() const { return i1 - i0 + 1; }
    int ny() const { return j1 - j0 + 1; }
};

// 将 (M-1) x (N-1) 的内部网格分成 Px x Py 个子块
class DomainDecomposer {
public:
    // M, N: 全局网格节点数（含边界），P: 总进程数 / 子域数
    DomainDecomposer(int M, int N, int P);

    int M()  const { return M_; }
    int N()  const { return N_; }
    int P()  const { return P_; }
    int Px() const { return Px_; }
    int Py() const { return Py_; }

    // 通过 rank 取子域（rank 在 0..P-1）
    Subdomain subdomain_for_rank(int rank) const;

    // 通过 (ix,iy) 取子域（ix 在 0..Px_-1, iy 在 0..Py_-1）
    Subdomain subdomain_at(int ix, int iy) const;

    // 打印分块信息（调试用）
    void print_summary(std::ostream& os) const;

    // 工具函数：选择 Px,Py，使 Px*Py=P，且子块尽量接近正方形
    static void choose_process_grid(int M, int N, int P, int& Px, int& Py);

    // 把 [1..nGlob] 分成 P 段，长度差 <= 1
    // 输出 cuts[0..P]，其中 cuts[0]=1, cuts[P]=nGlob+1
    static void split_1d(int nGlob, int P, std::vector<int>& cuts);

private:
    int M_;    // 全局网格节点数（含边界）
    int N_;
    int P_;    // 子域总数（= MPI 进程数）

    int Px_;   // x 方向子域数
    int Py_;   // y 方向子域数

    // 一维切分的端点：cutsX_[k] 是第 k 段的起始 i（内部索引，1-based）
    // 子域 ix 的 i 范围为 [cutsX_[ix], cutsX_[ix+1]-1]
    std::vector<int> cutsX_;
    std::vector<int> cutsY_;
};
