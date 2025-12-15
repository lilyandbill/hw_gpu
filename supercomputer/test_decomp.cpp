#include "domain_decomp.h"
#include <iostream>

int main() {
    int M = 40;
    int N = 40;

    for (int P : {1, 2, 4, 8}) {
        std::cout << "\n*** Decomposition for M=" << M
                  << ", N=" << N << ", P=" << P << " ***\n";
        DomainDecomposer dd(M, N, P);
        dd.print_summary(std::cout);
    }
    return 0;
}
