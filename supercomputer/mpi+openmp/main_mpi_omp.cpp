#include "solver_mpi_omp.h"
#include "domain_decomp.h"
#include <mpi.h>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc,char**argv){
    MPI_Init(&argc,&argv);

    int M=atoi(argv[1]);
    int N=atoi(argv[2]);

    MPI_Comm comm=MPI_COMM_WORLD;
    int rank,size;
    MPI_Comm_rank(comm,&rank);
    MPI_Comm_size(comm,&size);

    DomainDecomposer decomp(M,N,size);
    if(rank==0){
#ifdef _OPENMP
        std::cout<<"MPI ranks="<<size
                 <<" OMP threads="<<omp_get_max_threads()<<"\n";
#endif
        decomp.print_summary(std::cout);
    }

    MPISolver2D solver(M,N,comm,decomp);
    solver.build_ab_F_D();

    MPI_Barrier(comm);
    double t0=MPI_Wtime();
    int it=solver.pcg(1e-6,1e-6,100000);
    MPI_Barrier(comm);
    double t1=MPI_Wtime();

    if(rank==0)
        std::cout<<"iters="<<it<<" time="<<(t1-t0)<<" s\n";

    solver.gather_and_save("solution.csv");

    MPI_Finalize();
    return 0;
}
