#include "solver_mpi_omp.h"
#include <cmath>
#include <fstream>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

MPISolver2D::MPISolver2D(int M,int N,MPI_Comm comm,const DomainDecomposer& decomp)
: M_(M),N_(N),comm_(comm)
{
    MPI_Comm_rank(comm_,&rank_);
    MPI_Comm_size(comm_,&size_);
    sub_=decomp.subdomain_for_rank(rank_);

    nx_=sub_.nx();
    ny_=sub_.ny();
    halo_nx_=nx_+2;
    halo_ny_=ny_+2;

    h1_=(B1_-A1_)/M_;
    h2_=(B2_-A2_)/N_;
    h_=std::max(h1_,h2_);
    eps_=h_*h_;

    setup_neighbors(decomp);
}

void MPISolver2D::setup_neighbors(const DomainDecomposer& d){
    int Px=d.Px(), Py=d.Py();
    int ix=sub_.px, iy=sub_.py;
    nbr_left_   = (ix>0   )? iy*Px+ix-1:MPI_PROC_NULL;
    nbr_right_  = (ix<Px-1)? iy*Px+ix+1:MPI_PROC_NULL;
    nbr_bottom_ = (iy>0   )? (iy-1)*Px+ix:MPI_PROC_NULL;
    nbr_top_    = (iy<Py-1)? (iy+1)*Px+ix:MPI_PROC_NULL;
}

void MPISolver2D::build_ab_F_D(){
    F_.assign(nx_*ny_,0.0);
    Ddiag_.assign(nx_*ny_,4.0);

#pragma omp parallel for collapse(2)
    for(int j=0;j<ny_;++j)
        for(int i=0;i<nx_;++i){
            int gi=sub_.i0+i;
            int gj=sub_.j0+j;
            double x=A1_+gi*h1_, y=A2_+gj*h2_;
            F_[id_interior(i,j)] = (x>0 && y>0)?0.0:1.0;
        }
}

void MPISolver2D::exchange_halo(std::vector<double>& u) const{
    MPI_Request rq[8]; int k=0;
    std::vector<double> sL(ny_),sR(ny_),rL(ny_),rR(ny_);
    std::vector<double> sB(nx_),sT(nx_),rB(nx_),rT(nx_);

    for(int j=0;j<ny_;++j){
        sL[j]=u[id_halo(1,j+1)];
        sR[j]=u[id_halo(nx_,j+1)];
    }
    if(nbr_left_!=MPI_PROC_NULL){
        MPI_Irecv(rL.data(),ny_,MPI_DOUBLE,nbr_left_,0,comm_,&rq[k++]);
        MPI_Isend(sL.data(),ny_,MPI_DOUBLE,nbr_left_,1,comm_,&rq[k++]);
    }
    if(nbr_right_!=MPI_PROC_NULL){
        MPI_Irecv(rR.data(),ny_,MPI_DOUBLE,nbr_right_,1,comm_,&rq[k++]);
        MPI_Isend(sR.data(),ny_,MPI_DOUBLE,nbr_right_,0,comm_,&rq[k++]);
    }

    for(int i=0;i<nx_;++i){
        sB[i]=u[id_halo(i+1,1)];
        sT[i]=u[id_halo(i+1,ny_)];
    }
    if(nbr_bottom_!=MPI_PROC_NULL){
        MPI_Irecv(rB.data(),nx_,MPI_DOUBLE,nbr_bottom_,2,comm_,&rq[k++]);
        MPI_Isend(sB.data(),nx_,MPI_DOUBLE,nbr_bottom_,3,comm_,&rq[k++]);
    }
    if(nbr_top_!=MPI_PROC_NULL){
        MPI_Irecv(rT.data(),nx_,MPI_DOUBLE,nbr_top_,3,comm_,&rq[k++]);
        MPI_Isend(sT.data(),nx_,MPI_DOUBLE,nbr_top_,2,comm_,&rq[k++]);
    }
    MPI_Waitall(k,rq,MPI_STATUSES_IGNORE);

    if(nbr_left_!=MPI_PROC_NULL)
        for(int j=0;j<ny_;++j) u[id_halo(0,j+1)]=rL[j];
    if(nbr_right_!=MPI_PROC_NULL)
        for(int j=0;j<ny_;++j) u[id_halo(nx_+1,j+1)]=rR[j];
    if(nbr_bottom_!=MPI_PROC_NULL)
        for(int i=0;i<nx_;++i) u[id_halo(i+1,0)]=rB[i];
    if(nbr_top_!=MPI_PROC_NULL)
        for(int i=0;i<nx_;++i) u[id_halo(i+1,ny_+1)]=rT[i];
}

void MPISolver2D::apply_A(const std::vector<double>& u,
                          std::vector<double>& Au) const{
    std::vector<double> u2=u;
    exchange_halo(u2);
    Au.assign(halo_nx_*halo_ny_,0.0);

#pragma omp parallel for collapse(2)
    for(int j=0;j<ny_;++j)
        for(int i=0;i<nx_;++i){
            int id=id_halo(i+1,j+1);
            Au[id]=-4*u2[id]
                +u2[id_halo(i,j+1)]
                +u2[id_halo(i+2,j+1)]
                +u2[id_halo(i+1,j)]
                +u2[id_halo(i+1,j+2)];
        }
}

double MPISolver2D::dotE_local(const std::vector<double>& u,
                               const std::vector<double>& v) const{
    double s=0.0;
#pragma omp parallel for reduction(+:s) collapse(2)
    for(int j=0;j<ny_;++j)
        for(int i=0;i<nx_;++i){
            int id=id_halo(i+1,j+1);
            s+=u[id]*v[id];
        }
    return s;
}

double MPISolver2D::normE_global(const std::vector<double>& u) const{
    double loc=dotE_local(u,u), glob;
    MPI_Allreduce(&loc,&glob,1,MPI_DOUBLE,MPI_SUM,comm_);
    return std::sqrt(glob*h1_*h2_);
}

void MPISolver2D::solve_Dz_eq_r(const std::vector<double>& r,
                                std::vector<double>& z) const{
    z.assign(halo_nx_*halo_ny_,0.0);
#pragma omp parallel for collapse(2)
    for(int j=0;j<ny_;++j)
        for(int i=0;i<nx_;++i){
            z[id_halo(i+1,j+1)] =
                r[id_halo(i+1,j+1)] / Ddiag_[id_interior(i,j)];
        }
}

int MPISolver2D::pcg(double tol,double rtol,int maxIt,bool){
    w_.assign(halo_nx_*halo_ny_,0.0);
    r_.assign(halo_nx_*halo_ny_,0.0);
    p_.assign(halo_nx_*halo_ny_,0.0);
    Ap_.assign(halo_nx_*halo_ny_,0.0);
    z_.assign(halo_nx_*halo_ny_,0.0);
    z_new_.assign(halo_nx_*halo_ny_,0.0);

    for(int j=0;j<ny_;++j)
        for(int i=0;i<nx_;++i)
            r_[id_halo(i+1,j+1)] = F_[id_interior(i,j)];

    solve_Dz_eq_r(r_,z_);
    p_=z_;

    double zr_loc=dotE_local(z_,r_), zr;
    MPI_Allreduce(&zr_loc,&zr,1,MPI_DOUBLE,MPI_SUM,comm_);
    double normF=normE_global(r_);

    for(int it=0;it<maxIt;++it){
        apply_A(p_,Ap_);
        double pAp_loc=dotE_local(p_,Ap_), pAp;
        MPI_Allreduce(&pAp_loc,&pAp,1,MPI_DOUBLE,MPI_SUM,comm_);
        double alpha=zr/pAp;

#pragma omp parallel for collapse(2)
        for(int j=0;j<ny_;++j)
            for(int i=0;i<nx_;++i){
                int id=id_halo(i+1,j+1);
                w_[id]+=alpha*p_[id];
                r_[id]-=alpha*Ap_[id];
            }

        double nr=normE_global(r_);
        if(nr<tol || nr<rtol*normF) return it+1;

        solve_Dz_eq_r(r_,z_new_);
        double zr_new_loc=dotE_local(z_new_,r_), zr_new;
        MPI_Allreduce(&zr_new_loc,&zr_new,1,MPI_DOUBLE,MPI_SUM,comm_);
        double beta=zr_new/zr; zr=zr_new;

#pragma omp parallel for collapse(2)
        for(int j=0;j<ny_;++j)
            for(int i=0;i<nx_;++i){
                int id=id_halo(i+1,j+1);
                p_[id]=z_new_[id]+beta*p_[id];
                z_[id]=z_new_[id];
            }
    }
    return maxIt;
}

void MPISolver2D::gather_and_save(const std::string& f) const{
    int K=nx_*ny_;
    std::vector<double> loc(K);
    for(int j=0;j<ny_;++j)
        for(int i=0;i<nx_;++i)
            loc[id_interior(i,j)] = w_[id_halo(i+1,j+1)];

    std::vector<int> cnt(size_),disp(size_);
    MPI_Gather(&K,1,MPI_INT,cnt.data(),1,MPI_INT,0,comm_);
    if(rank_==0){
        disp[0]=0;
        for(int i=1;i<size_;++i) disp[i]=disp[i-1]+cnt[i-1];
    }
    std::vector<double> all;
    if(rank_==0) all.resize(disp.back()+cnt.back());
    MPI_Gatherv(loc.data(),K,MPI_DOUBLE,
                all.data(),cnt.data(),disp.data(),MPI_DOUBLE,0,comm_);
    if(rank_==0){
        std::ofstream ofs(f);
        for(double v:all) ofs<<v<<"\n";
    }
}
