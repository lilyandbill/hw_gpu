#include "solver.h"
#include <iostream>
#include <iomanip>
#include <chrono>

int main() {
    const double tol   = 1e-8;   
    const double rtol  = 1e-10;  // 相对残差
    const int    maxIt = -1;     

    auto run_case = [&](int M, int N) {
        std::cout << "\n=== Running case M=" << M << ", N=" << N << " (serial) ===\n";

        Solver2D solver(M, N);
        solver.build_ab();
        solver.build_F();
        solver.build_Ddiag();

        std::vector<double> w;
        auto t0 = std::chrono::high_resolution_clock::now();
        int iters = solver.pcg(w, tol, rtol, maxIt, /*verbose=*/false);
        auto t1 = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> dt = t1 - t0;
        std::cout << "iters = " << iters
                  << ", time = " << std::fixed << std::setprecision(6) << dt.count() << " s\n";

        std::string fname = "solution_" + std::to_string(M) + "x" + std::to_string(N) + ".csv";
        solver.save_csv(fname, w);
        std::cout << "Saved: " << fname << "\n";
    };

    run_case(10, 10);
    run_case(20, 20);
    run_case(40, 40);

    std::cout << "\nAll cases finished (serial build).\n";
    return 0;
}
