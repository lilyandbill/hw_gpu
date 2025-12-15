#include "solver_openmp.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

static void print_stats(const RunStats& s) {
    std::cout << "  T=" << std::setw(2) << s.threads
              << " | iters=" << std::setw(4) << s.iters
              << " | build=" << std::fixed << std::setprecision(6) << s.t_build << "s"
              << " | solve=" << s.t_solve << "s"
              << " | total=" << s.t_total << "s";
    if (!s.saved_csv.empty()) std::cout << " | saved=" << s.saved_csv;
    std::cout << "\n";
}

int main() {
    const int M = 800, N = 1200;     
    const double tol  = 1e-8;     
    const double rtol = 1e-10;    
    const bool saveCsv = true;

#ifdef _OPENMP
    std::cout << "[OpenMP build detected]\n";
#else
    std::cout << "[Serial build (no OpenMP pragmas)]\n";
#endif

    TimedSolverDecorator runner;

    std::cout << "=== Run on grid " << M << "x" << N << " ===\n";
    std::vector<int> thread_list = {1,4,16};  

    // T=1
    std::vector<RunStats> all;
    for (int T : thread_list) {
        auto s = runner.run(M, N, T, tol, rtol, saveCsv, "solution");
        all.push_back(s);
        print_stats(s);
    }

    // T=1  total as standard 
    if (!all.empty()) {
        double T1 = all.front().t_total;
        std::cout << "\nSpeedup relative to T=1:\n";
        for (const auto& s : all) {
            double Sp = T1 / s.t_total;
            std::cout << "  T=" << std::setw(2) << s.threads
                      << " | speedup=" << std::fixed << std::setprecision(3) << Sp << "x\n";
        }
    }

    std::cout << "\nDone.\n";
    return 0;
}
