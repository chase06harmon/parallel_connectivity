#include "benchmarks/Connectivity/Gazit1991/Connectivity.h"

namespace gbbs {
namespace gazit_cc {

template <class Graph>
double GazitRunner(Graph& G, commandLine P) {
  GazitParams params;

    params.alpha = P.getOptionDoubleValue("-alpha", params.alpha);
    params.processor_budget =
        static_cast<size_t>(P.getOptionLongValue("-processor_budget", params.processor_budget));
    params.max_rounds =
        static_cast<size_t>(P.getOptionLongValue("-max_rounds", params.max_rounds));
    params.seed = static_cast<uint64_t>(P.getOptionLongValue("-seed", params.seed));
  
    double elapsed;
  
    std::cout << "### Application: GazitCC" << std::endl;
    std::cout << "### Graph: " << P.getArgument(0) << std::endl;
    std::cout << "### Threads: " << num_workers() << std::endl;
    std::cout << "### n: " << G.n << std::endl;
    std::cout << "### m: " << G.m << std::endl;
    std::cout << "### Params: -alpha = " << params.alpha
              << " -processor_budget = " << params.processor_budget
              << " -max_rounds = " << params.max_rounds
              << " -seed = " << params.seed
              << " -skip_sparse_to_dense = " << params.skip_sparse_to_dense
              << std::endl;
    std::cout << "### ------------------------------------" << std::endl;

    timer t;
    t.start();
    CC_eval(G, params);
    params.skip_sparse_to_dense = true;

    CC_eval(G, params);

    params.easy_case_only = true;

    CC_eval(G, params);

    elapsed = t.stop();

    std::cout << "### Running Time: " << elapsed << std::endl;
    if (P.getOption("-stats")) {
      std::cout << "# (stats collection not implemented yet)" << std::endl;
    }


  return elapsed;
}

}  // namespace gazit_cc
}  // namespace gbbs

generate_symmetric_main(gbbs::gazit_cc::GazitRunner, false);
