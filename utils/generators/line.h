#pragma once

#include "gbbs/bridge.h"
#include "gbbs/macros.h"

namespace gbbs {
namespace line {

auto generate_updates(size_t n) {
  // For a symmetric graph, we generate edges in canonical form (u < v)
  // Each vertex will pick k neighbors, but we store each unique edge only once
  size_t m = n-1;
  auto edges = parlay::sequence<std::pair<uintE, uintE>>(m);
  
  parlay::random rnd;
  parallel_for(0, m, [&](size_t i) {
    edges[i].first = i; 
    edges[i].second = i + 1 % n;
  });
  
  return edges;
}

}  // namespace regular_random
}  // namespace gbbs
