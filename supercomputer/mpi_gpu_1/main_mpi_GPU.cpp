#include "solver_mpi_GPU.h"
#include "domain_decomp_GPU.h"

#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <cstdlib>


#include <cuda_runtime.h>


int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0, size = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // 默认网格大小，可以用命令行改：mpirun -n P ./a.out M N
    int M = 40, N = 40;
    if (argc >= 3) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
    }

    double tol  = 1e-3;
    double rtol = 1e-3;

    // 简单的 rank -> GPU 映射：第 r 个进程用 (r % deviceCount) 号 GPU
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount > 0) {
        int dev = rank % deviceCount;
        cudaSetDevice(dev);
        if (rank == 0) {
            std::cout << "Detected " << deviceCount
                      << " CUDA devices, using dev = rank % "
                      << deviceCount << std::endl;
        }
    } else if (rank == 0) {
        std::cerr << "Warning: no CUDA device found, running on CPU.\n";
    }

    if (rank == 0) {
        std::cout << "Problem size: M=" << M
                  << ", N=" << N
                  << ", MPI processes P=" << size << std::endl;
    }

    // ========= 域分解 =========
    DomainDecomposer decomp(M, N, size);
    if (rank == 0) {
        decomp.print_summary(std::cout);  // 如果不想打印可以删掉这一行
    }

    // ========= 构造 GPU 版求解器 =========
    // 假定 GPU 版的实现仍然叫 MPISolver2D，只是实现放在 solver_mpi_GPU.* 里
    MPISolver2D solver(M, N, comm, decomp);

    // ========= 组装阶段（矩阵+右端项） =========
    MPI_Barrier(comm);
    double t_build0 = MPI_Wtime();
    solver.build_ab_F_D();     // 这里面你可以在 solver 里做 GPU 数据分配/拷贝
    MPI_Barrier(comm);
    double t_build1 = MPI_Wtime();

    // ========= 求解阶段（PCG） =========
    double t_solve0 = MPI_Wtime();
    int iters = solver.pcg(tol, rtol, 10000, false);
    MPI_Barrier(comm);
    double t_solve1 = MPI_Wtime();

    // ========= 统计时间（按各进程最大值） =========
    double t_build_local = t_build1 - t_build0;
    double t_solve_local = t_solve1 - t_solve0;
    double t_total_local = t_build_local + t_solve_local;

    double t_build = 0.0, t_solve = 0.0, t_total = 0.0;
    MPI_Reduce(&t_build_local, &t_build, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&t_solve_local, &t_solve, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&t_total_local, &t_total, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (rank == 0) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "P=" << size
                  << "  iters=" << iters << "\n"
                  << "  build time (max over ranks): " << t_build  << " s\n"
                  << "  solve time (max over ranks): " << t_solve  << " s\n"
                  << "  total time (max over ranks): " << t_total  << " s\n";
    }

    // ========= 收集并输出解 =========
    solver.gather_and_save("solution_P" + std::to_string(size) + "_GPU.csv");

    MPI_Finalize();
    return 0;
}
