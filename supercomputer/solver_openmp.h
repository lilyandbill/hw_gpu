#pragma once
#include <vector>
#include <string>
#ifdef _OPENMP
  #include <omp.h>
#endif


class Solver2D {
public:
    explicit Solver2D(int M_, int N_);

    void build_ab();     
    void build_F();      
    void build_Ddiag();  
    void apply_A(const std::vector<double>& x, std::vector<double>& y) const;

    double dotE(const std::vector<double>& u, const std::vector<double>& v) const;
    double normE(const std::vector<double>& u) const;
    void solve_Dz_eq_r(const std::vector<double>& r, std::vector<double>& z) const;

    int pcg(std::vector<double>& w, double tol=1e-8, double rtol=1e-10,
            int maxIt=-1, bool verbose=false);

    void save_csv(const std::string& fname, const std::vector<double>& w) const;

    int M() const { return M_; }
    int N() const { return N_; }

private:
    int M_, N_;
    double A1_=-1.0, B1_= 1.0;
    double A2_=-1.0, B2_= 1.0;
    double h1_, h2_, h_, eps_;

    std::vector<double> a_, b_, F_, Ddiag_;

    inline int idxU (int i, int j) const { return (j-1)*(M_-1) + (i-1); }   // 内部未知
    inline int idxAB(int i, int j) const { return i*(N_+1) + j; }           // 边系数
    inline bool isInterior(int i, int j) const { return (i>=1 && i<=M_-1 && j>=1 && j<=N_-1); }
};



struct RunStats {
    int M=0, N=0, threads=1, iters=0;
    double t_build=0.0;   // a/b/F/Ddiag
    double t_solve=0.0;   // PCG time
    double t_total=0.0;   
    double tol=0.0, rtol=0.0;
    std::string saved_csv;
};

class TimedSolverDecorator {
public:
    
    RunStats run(int M, int N, int threads, double tol, double rtol,
                 bool saveCsv, const std::string& csvNameBase) const;
};
