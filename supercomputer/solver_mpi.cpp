#include "solver_mpi.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <mpi.h>

using namespace std;

// =======================================
//          构造函数与邻居设置
// =======================================
MPISolver2D::MPISolver2D(int M, int N, MPI_Comm comm, const DomainDecomposer& decomp)
    : M_(M), N_(N), comm_(comm)
{
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &size_);
    sub_ = decomp.subdomain_for_rank(rank_);
    nx_ = sub_.nx();
    ny_ = sub_.ny();
    halo_nx_ = nx_ + 2;
    halo_ny_ = ny_ + 2;

    h1_ = (B1_ - A1_) / M_;
    h2_ = (B2_ - A2_) / N_;
    h_  = max(h1_, h2_);
    eps_ = h_ * h_;

    setup_neighbors(decomp);
}

// 邻居关系：仅用于笛卡尔通信
void MPISolver2D::setup_neighbors(const DomainDecomposer& decomp)
{
    int Px = decomp.Px();
    int Py = decomp.Py();
    int ix = sub_.px, iy = sub_.py;

    nbr_left_   = (ix > 0)     ? iy*Px + (ix-1) : MPI_PROC_NULL;
    nbr_right_  = (ix < Px-1)  ? iy*Px + (ix+1) : MPI_PROC_NULL;
    nbr_bottom_ = (iy > 0)     ? (iy-1)*Px + ix : MPI_PROC_NULL;
    nbr_top_    = (iy < Py-1)  ? (iy+1)*Px + ix : MPI_PROC_NULL;
}

// =======================================
//          系数矩阵与源项构建
// =======================================
void MPISolver2D::build_ab_F_D()
{
    aL_.assign(nx_*ny_, 1.0);
    aR_.assign(nx_*ny_, 1.0);
    bB_.assign(nx_*ny_, 1.0);
    bT_.assign(nx_*ny_, 1.0);
    F_.assign(nx_*ny_, 0.0);
    Ddiag_.assign(nx_*ny_, 0.0);

    for (int jj=0;jj<ny_;++jj){
        int j=sub_.j0+jj;
        for (int ii=0;ii<nx_;++ii){
            int i=sub_.i0+ii;
            double x=A1_+i*h1_, y=A2_+j*h2_;
            double k=1.0;
            if(x>0 && y>0) k=1.0/eps_; // 靴子外区域
            F_[id_interior(ii,jj)]= (k==1.0)?1.0:0.0;
            Ddiag_[id_interior(ii,jj)]=4.0;
        }
    }
}

// =======================================
//          Halo 交换函数
// =======================================
void MPISolver2D::exchange_halo(vector<double>& u) const
{
    MPI_Request reqs[8];
    int rc=0;

    vector<double> sendL(ny_),sendR(ny_),recvL(ny_),recvR(ny_);
    for(int jj=0;jj<ny_;++jj){
        sendL[jj]=u[id_halo(1,jj+1)];
        sendR[jj]=u[id_halo(nx_,jj+1)];
    }
    if(nbr_left_!=MPI_PROC_NULL){
        MPI_Irecv(recvL.data(),ny_,MPI_DOUBLE,nbr_left_,0,comm_,&reqs[rc++]);
        MPI_Isend(sendL.data(),ny_,MPI_DOUBLE,nbr_left_,1,comm_,&reqs[rc++]);
    }
    if(nbr_right_!=MPI_PROC_NULL){
        MPI_Irecv(recvR.data(),ny_,MPI_DOUBLE,nbr_right_,1,comm_,&reqs[rc++]);
        MPI_Isend(sendR.data(),ny_,MPI_DOUBLE,nbr_right_,0,comm_,&reqs[rc++]);
    }

    vector<double> sendB(nx_),sendT(nx_),recvB(nx_),recvT(nx_);
    for(int ii=0;ii<nx_;++ii){
        sendB[ii]=u[id_halo(ii+1,1)];
        sendT[ii]=u[id_halo(ii+1,ny_)];
    }
    if(nbr_bottom_!=MPI_PROC_NULL){
        MPI_Irecv(recvB.data(),nx_,MPI_DOUBLE,nbr_bottom_,2,comm_,&reqs[rc++]);
        MPI_Isend(sendB.data(),nx_,MPI_DOUBLE,nbr_bottom_,3,comm_,&reqs[rc++]);
    }
    if(nbr_top_!=MPI_PROC_NULL){
        MPI_Irecv(recvT.data(),nx_,MPI_DOUBLE,nbr_top_,3,comm_,&reqs[rc++]);
        MPI_Isend(sendT.data(),nx_,MPI_DOUBLE,nbr_top_,2,comm_,&reqs[rc++]);
    }

    MPI_Waitall(rc,reqs,MPI_STATUSES_IGNORE);

    if(nbr_left_!=MPI_PROC_NULL)
        for(int jj=0;jj<ny_;++jj) u[id_halo(0,jj+1)]=recvL[jj];
    if(nbr_right_!=MPI_PROC_NULL)
        for(int jj=0;jj<ny_;++jj) u[id_halo(nx_+1,jj+1)]=recvR[jj];
    if(nbr_bottom_!=MPI_PROC_NULL)
        for(int ii=0;ii<nx_;++ii) u[id_halo(ii+1,0)]=recvB[ii];
    if(nbr_top_!=MPI_PROC_NULL)
        for(int ii=0;ii<nx_;++ii) u[id_halo(ii+1,ny_+1)]=recvT[ii];
}

// =======================================
//          应用矩阵 A
// =======================================
void MPISolver2D::apply_A(const std::vector<double>& u, std::vector<double>& Au) const
{
    // 先把 u 拷贝一份用于 halo 交换
    std::vector<double> u2 = u;
    exchange_halo(u2);

    // 输出向量也按 halo 尺寸来，外圈先清零
    Au.assign(halo_nx_ * halo_ny_, 0.0);

    for (int jj = 0; jj < ny_; ++jj) {
        for (int ii = 0; ii < nx_; ++ii) {
            // 中心和四邻居都用 halo 索引
            double wC = u2[id_halo(ii + 1,     jj + 1)];
            double wL = u2[id_halo(ii,         jj + 1)];
            double wR = u2[id_halo(ii + 2,     jj + 1)];
            double wB = u2[id_halo(ii + 1,     jj    )];
            double wT = u2[id_halo(ii + 1,     jj + 2)];

            // 把 A*u 的结果写回到 Au 的内部格点 (同样用 halo 索引)
            Au[id_halo(ii + 1, jj + 1)] = (-4.0 * wC + wL + wR + wB + wT);
        }
    }
}


// =======================================
//          向量运算
// =======================================
double MPISolver2D::dotE_local(const vector<double>& u,const vector<double>& v) const{
    double s=0.0;
    for(int jj=0;jj<ny_;++jj)
        for(int ii=0;ii<nx_;++ii)
            s+=u[id_halo(ii+1,jj+1)]*v[id_halo(ii+1,jj+1)];
    return s;
}
double MPISolver2D::normE_global(const vector<double>& u) const{
    double loc=dotE_local(u,u);
    double glob;
    MPI_Allreduce(&loc,&glob,1,MPI_DOUBLE,MPI_SUM,comm_);
    return sqrt(glob*h1_*h2_);
}
void MPISolver2D::solve_Dz_eq_r(const vector<double>& r, vector<double>& z) const{
    z.assign(halo_nx_*halo_ny_,0.0);
    for(int jj=0;jj<ny_;++jj)
        for(int ii=0;ii<nx_;++ii)
            z[id_halo(ii+1,jj+1)] = r[id_halo(ii+1,jj+1)] / Ddiag_[id_interior(ii,jj)];
}

// =======================================
//          PCG 主循环
// =======================================
int MPISolver2D::pcg(double tol,double rtol,int maxIt,bool verbose)
{
    int Kloc=nx_*ny_;
    w_.assign(halo_nx_*halo_ny_,0.0);
    r_.assign(halo_nx_*halo_ny_,0.0);
    p_.assign(halo_nx_*halo_ny_,0.0);
    Ap_.assign(halo_nx_*halo_ny_,0.0);
    z_.assign(halo_nx_*halo_ny_,0.0);
    z_new_.assign(halo_nx_*halo_ny_,0.0);

    // 初始 r = F, z = D^{-1}r, p=z
    for(int jj=0;jj<ny_;++jj)
        for(int ii=0;ii<nx_;++ii)
            r_[id_halo(ii+1,jj+1)] = F_[id_interior(ii,jj)];
    solve_Dz_eq_r(r_,z_);
    p_=z_;

    double zr_loc=dotE_local(z_,r_), zr_glob;
    MPI_Allreduce(&zr_loc,&zr_glob,1,MPI_DOUBLE,MPI_SUM,comm_);

    double normF=normE_global(r_);
    if(normF<1e-3) normF=1.0;
    int it=0;

    for(;it<100000;++it){
        apply_A(p_,Ap_);
        double pAp_loc=dotE_local(p_,Ap_), pAp_glob;
        MPI_Allreduce(&pAp_loc,&pAp_glob,1,MPI_DOUBLE,MPI_SUM,comm_);
        double alpha=zr_glob/pAp_glob;

        // w+=alpha*p, r-=alpha*Ap
        for(int jj=0;jj<ny_;++jj)
            for(int ii=0;ii<nx_;++ii){
                int idh=id_halo(ii+1,jj+1);
                w_[idh]+=alpha*p_[idh];
                r_[idh]-=alpha*Ap_[idh];
            }

        // double nr_loc=dotE_local(r_,r_), nr_glob;
        // MPI_Allreduce(&nr_loc,&nr_glob,1,MPI_DOUBLE,MPI_SUM,comm_);
        // double nr=sqrt(nr_glob*h1_*h2_);
        // if(nr <= tol || nr <= rtol * normF) break;
        double nr_loc = dotE_local(r_, r_);
        double nr_glob;
        MPI_Allreduce(&nr_loc, &nr_glob, 1, MPI_DOUBLE, MPI_SUM, comm_);
        double nr = sqrt(nr_glob * h1_ * h2_);

        // 收敛条件：
        if (nr <= tol) break;               // 绝对残差
        if (nr <= rtol * normF) break;      // 相对残差
        if (nr <= 1e-12) break;             // 防止浮点误差

        solve_Dz_eq_r(r_,z_new_);
        double zr_new_loc=dotE_local(z_new_,r_), zr_new_glob;
        MPI_Allreduce(&zr_new_loc,&zr_new_glob,1,MPI_DOUBLE,MPI_SUM,comm_);
        double beta=zr_new_glob/zr_glob;
        zr_glob=zr_new_glob;

        for(int jj=0;jj<ny_;++jj)
            for(int ii=0;ii<nx_;++ii){
                int idh=id_halo(ii+1,jj+1);
                p_[idh]=z_new_[idh]+beta*p_[idh];
                z_[idh]=z_new_[idh];
            }
    }
    return it;
}

// =======================================
//          收集并保存结果
// =======================================
void MPISolver2D::gather_and_save(const string& fname) const
{
    int Kloc=nx_*ny_;
    vector<double> loc(Kloc);
    for(int jj=0;jj<ny_;++jj)
        for(int ii=0;ii<nx_;++ii)
            loc[id_interior(ii,jj)] = w_[id_halo(ii+1,jj+1)];

    int locCount=Kloc;
    vector<int> counts(size_), displs(size_);
    MPI_Gather(&locCount,1,MPI_INT,counts.data(),1,MPI_INT,0,comm_);

    if(rank_==0){
        displs[0]=0;
        for(int i=1;i<size_;++i) displs[i]=displs[i-1]+counts[i-1];
    }

    vector<double> global;
    int totalCount=0;
    if(rank_==0){
        totalCount=displs[size_-1]+counts[size_-1];
        global.resize(totalCount);
    }

    MPI_Gatherv(loc.data(),locCount,MPI_DOUBLE,
                global.data(),counts.data(),displs.data(),MPI_DOUBLE,0,comm_);

    if(rank_==0){
        ofstream f(fname);
        for(double v:global) f<<v<<"\n";
        cerr<<"Saved solution to "<<fname<<"\n";
    }
}
