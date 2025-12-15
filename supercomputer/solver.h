#pragma once
#include <vector>
#include <string>

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
    double h1() const { return h1_; }
    double h2() const { return h2_; }

private:
    int M_, N_;
    double A1_=-1.0, B1_= 1.0;   // x 区间
    double A2_=-1.0, B2_= 1.0;   // y 区间
    double h1_, h2_, h_, eps_;

    std::vector<double> a_, b_;
    std::vector<double> F_;
    std::vector<double> Ddiag_;


    inline int idxU(int i, int j) const { return (j-1)*(M_-1) + (i-1); }         // 内部 
    inline int idxAB(int i, int j) const { return i*(N_+1) + j; }                // a/b (1..M,1..N)


    inline bool isInterior(int i, int j) const { return (i>=1 && i<=M_-1 && j>=1 && j<=N_-1); }
};
