#include "benchmarks/Connectivity/Gazit1991/Connectivity_Tests.h"

namespace gbbs {
namespace gazit_cc {

template<class Graph>
double TestRunner(Graph& G, commandLine P) {
    std::cout << "### Application: GazitCC" << std::endl;
    std::cout << "### Graph: " << P.getArgument(0) << std::endl;
    std::cout << "### Threads: " << num_workers() << std::endl;
    std::cout << "### n: " << G.n << std::endl;
    std::cout << "### m: " << G.m << std::endl;
    // std::cout << "### ------------------------------------" << std::endl;

    timer t;
    t.start();
    // test_sample_edges(G);
    // test_paritioning(G);
    test_sparse_to_dense(G);
    double elapsed = t.stop();

    std::cout << "### Running Time: " << elapsed << std::endl;
    if (P.getOption("-stats")) {
        std::cout << "# (stats collection not implemented yet)" << std::endl;
    }

    std::cout << "### ------------------------------------" << std::endl << std::endl;
    
    return elapsed;
}

}
}


generate_symmetric_main(gbbs::gazit_cc::TestRunner, false);