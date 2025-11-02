#include "benchmarks/Connectivity/Gazit1991/Connectivity.h"

namespace gbbs {
namespace gazit_cc {

template <class Graph>
double GazitRunner(Graph& G, commandLine P) {
  GazitParams params;
  params.alpha = P.getOptionDoubleValue("-alpha", params.alpha);
  params.processor_budget =
      static_cast<size_t>(P.getOptionLongValue("-processor_budget", 0));
  params.max_rounds =
      static_cast<size_t>(P.getOptionLongValue("-max_rounds", 0));
  params.seed = static_cast<uint64_t>(P.getOptionLongValue("-seed", 5489));

  std::cout << "### Application: GazitCC" << std::endl;
  std::cout << "### Graph: " << P.getArgument(0) << std::endl;
  std::cout << "### Threads: " << num_workers() << std::endl;
  std::cout << "### n: " << G.n << std::endl;
  std::cout << "### m: " << G.m << std::endl;
  std::cout << "### Params: -alpha = " << params.alpha
            << " -processor_budget = " << params.processor_budget
            << " -max_rounds = " << params.max_rounds
            << " -seed = " << params.seed << std::endl;
  std::cout << "### ------------------------------------" << std::endl;

  timer t;
  t.start();
  auto components = CC(G, params);
  double elapsed = t.stop();
  components.clear();

  std::cout << "### Running Time: " << elapsed << std::endl;
  if (P.getOption("-stats")) {
    std::cout << "# (stats collection not implemented yet)" << std::endl;
  }
  return elapsed;
}

}  // namespace gazit_cc
}  // namespace gbbs

generate_symmetric_main(gbbs::gazit_cc::GazitRunner, false);
