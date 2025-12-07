#include "rmat.h"

#include "gbbs/gbbs.h"
#include "gbbs/graph.h"
#include "gbbs/graph_io.h"

#include <fstream>
#include <iostream>

namespace gbbs {

using Edge = std::pair<uintE, uintE>;

int BuildRegularRandom(int argc, char* argv[]) {
  commandLine P(argc, argv, "");

  size_t n = P.getOptionLongValue("-n", 1UL << 27);
  size_t m = P.getOptionLongValue("-m", 500000000);

  double a = P.getOptionDoubleValue("-a", 0.25);
  double b = P.getOptionDoubleValue("-b", 0.25);
  double c = P.getOptionDoubleValue("-c", 0.25);

  auto out_f = P.getOptionValue("-outfile", "");

  if (out_f == "") {
    std::cout << "specify a valid outfile using -outfile" << std::endl;
    abort();
  }

  uintE seed = 4;
  std::cout << "Generating updates" << std::endl;
  auto updates = rmat::generate_updates(n, m, seed, a, b, c);
  std::cout << "Generated updates" << std::endl;

  // Convert to Edge format
  sequence<std::tuple<unsigned int, unsigned int, gbbs::empty>> edge_list(updates.size());
  parallel_for(0, updates.size(), [&](size_t i) {
    std::get<0>(edge_list[i]) = std::get<0>(updates[i]);
    std::get<1>(edge_list[i]) = std::get<1>(updates[i]);
    std::get<2>(edge_list[i]) = gbbs::empty{};
  });

  // Build symmetric (undirected) graph from edges
  std::cout << "Building symmetric graph..." << std::endl;
  auto G = symmetric_graph<symmetric_vertex, gbbs::empty>::from_edges(edge_list, n);
  std::cout << "Graph built: " << G.n << " vertices, " << G.m << " edges" << std::endl;

  // Write graph in CSR (AdjacencyGraph) format
  std::cout << "Writing graph to " << out_f << " in CSR format..." << std::endl;
  gbbs_io::write_graph_to_file(out_f.c_str(), G);

  std::cout << "done" << std::endl;
  return 0;
}

}  // namespace gbbs

int main(int argc, char* argv[]) {
  return gbbs::BuildRegularRandom(argc, argv);
}
