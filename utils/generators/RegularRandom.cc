#include "regular_random.h"

#include "gbbs/gbbs.h"
#include "gbbs/graph.h"
#include "gbbs/graph_io.h"

#include <fstream>
#include <iostream>

namespace gbbs {

using Edge = std::pair<uintE, uintE>;

int BuildRegularRandom(int argc, char* argv[]) {
  commandLine P(argc, argv, "");

  size_t n = P.getOptionLongValue("-n", 1UL << 20);
  size_t k = P.getOptionLongValue("-k", 5);

  auto out_f = P.getOptionValue("-outfile", "");

  if (out_f == "") {
    std::cout << "specify a valid outfile using -outfile" << std::endl;
    abort();
  }

  std::cout << "Generating regular random graph with n=" << n 
            << " vertices, k=" << k << " edges per vertex" << std::endl;
  
  // Generate edge list
  auto updates = regular_random::generate_updates(n, k);
  std::cout << "Generated " << updates.size() << " edges" << std::endl;

  // Convert to Edge format
  sequence<std::tuple<unsigned int, unsigned int, gbbs::empty>> edge_list(updates.size());
  parallel_for(0, updates.size(), [&](size_t i) {
    edge_list[i] = {updates[i].first, updates[i].second, gbbs::empty()};
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
