#include "benchmarks/Connectivity/Gazit1991/Connectivity.h"

namespace gbbs {
namespace gazit_cc {

template <class Graph>
double ComparisonRunner(Graph& G, commandLine P) {
  double beta = P.getOptionDoubleValue("-beta", 0.2);
  bool permute = P.getOption("-permute");

  GazitParams params;
  params.alpha = P.getOptionDoubleValue("-alpha", params.alpha);
  params.processor_budget =
      static_cast<size_t>(P.getOptionLongValue("-processor_budget", 0));
  params.max_rounds =
      static_cast<size_t>(P.getOptionLongValue("-max_rounds", 0));
  params.seed = static_cast<uint64_t>(P.getOptionLongValue("-seed", 5489));

  auto stats = BenchmarkPair(G, beta, permute, params);

  std::cout << "### Comparison: WorkEfficientSDB14 vs Gazit" << std::endl;
  std::cout << "### Graph: " << P.getArgument(0) << std::endl;
  std::cout << "### Threads: " << num_workers() << std::endl;
  std::cout << "### n: " << G.n << std::endl;
  std::cout << "### m: " << G.m << std::endl;
  std::cout << "### Params: -beta = " << beta
            << " -permute = " << permute << " -alpha = " << params.alpha
            << " -processor_budget = " << params.processor_budget
            << " -max_rounds = " << params.max_rounds
            << " -seed = " << params.seed << std::endl;
  std::cout << "### ------------------------------------" << std::endl;
  std::cout << "# workefficient_time = " << stats.workefficient_time << std::endl;
  std::cout << "# gazit_time = " << stats.gazit_time << std::endl;

  // Return Gazit time so the harness reports the stub runtime.
  return stats.gazit_time;
}

}  // namespace gazit_cc
}  // namespace gbbs

generate_symmetric_main(gbbs::gazit_cc::ComparisonRunner, false);
