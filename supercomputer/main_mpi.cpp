#include "solver_mpi.h"
#include "domain_decomp.h"
#include <mpi.h>
#include <iostream>
#include <iomanip>

int main(int argc,char**argv){
    MPI_Init(&argc,&argv);

    int M=800,N=1200;
    M = atoi(argv[1]);
    N = atoi(argv[2]);
    double tol=1e-3,rtol=1e-3;
    MPI_Comm comm=MPI_COMM_WORLD;
    int rank,size;
    MPI_Comm_rank(comm,&rank);
    MPI_Comm_size(comm,&size);

    DomainDecomposer decomp(M,N,size);
    if(rank==0) decomp.print_summary(std::cout);

    MPISolver2D solver(M,N,comm,decomp);
    solver.build_ab_F_D();

    MPI_Barrier(comm);
    double t0=MPI_Wtime();
    int iters=solver.pcg(tol,rtol,100000,false);
    MPI_Barrier(comm);
    double t1=MPI_Wtime();
    double local=t1-t0,global;
    MPI_Reduce(&local,&global,1,MPI_DOUBLE,MPI_MAX,0,comm);

    if(rank==0)
        std::cout<<"P="<<size<<" iters="<<iters<<" time="<<global<<" s\n";
    solver.gather_and_save("solution_P"+std::to_string(size)+".csv");

    MPI_Finalize();
    return 0;
}
