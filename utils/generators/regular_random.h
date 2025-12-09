#pragma once

#include "gbbs/bridge.h"
#include "gbbs/macros.h"

namespace gbbs {
namespace regular_random {

// Generates a symmetric random regular graph where each vertex has exactly k edges
// to randomly chosen neighbors. Returns edges as pairs (u, v) where u < v
// (canonical form). The symmetric_graph::from_edges() will automatically
// create both (u,v) and (v,u) when building the graph.
auto generate_updates(size_t n, size_t k = 5) {
  // For a symmetric graph, we generate edges in canonical form (u < v)
  // Each vertex will pick k neighbors, but we store each unique edge only once
  size_t m = n * k;
  auto edges = parlay::sequence<std::pair<uintE, uintE>>(m);
  
  parlay::random rnd;
  parallel_for(0, n, [&](size_t i) {
    auto i_rnd = rnd.fork(i);
    
    // Generate k random neighbors for vertex i
    for (size_t j = 0; j < k; j++) {
      uintE neighbor;
      bool valid = false;
      
      // Keep trying until we get a valid neighbor (not self, not duplicate)
      while (!valid) {
        neighbor = i_rnd.rand() % n;
        
        // Check if it's not self and not already chosen
        if (neighbor != i) {
          valid = true;
          // Check for duplicates in previously chosen neighbors
          for (size_t prev = 0; prev < j; prev++) {
            if (edges[i * k + prev].second == neighbor) {
              valid = false;
              break;
            }
          }
        }
        
        if (!valid) {
          i_rnd = i_rnd.next();
        }
      }
      
      // Store edge in canonical form (smaller vertex first)
      // This ensures symmetry when the graph is built
      if (i < neighbor) {
        edges[i * k + j] = std::make_pair(i, neighbor);
      } else {
        edges[i * k + j] = std::make_pair(neighbor, i);
      }
      i_rnd = i_rnd.next();
    }
  });
  
  return edges;
}

}  // namespace regular_random
}  // namespace gbbs
