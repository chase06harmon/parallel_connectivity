
#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <atomic>

#include "benchmarks/Connectivity/WorkEfficientSDB14/Connectivity.h"
#include "benchmarks/Connectivity/common.h"
#include "gbbs/gbbs.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

namespace gbbs {
namespace gazit_cc {

struct GazitParams {
  double alpha = 1.5;
  size_t processor_budget = 0;
  size_t max_rounds = 0;
  uint64_t seed = 5489;
  bool deduplicate_edges = false;
};

struct ComparisonStats {
  double workefficient_time = 0.0;
  double gazit_time = 0.0;
};

namespace internal {

using Edge = std::pair<uintE, uintE>;

inline uintE find_root(sequence<parent>& parents, uintE v) {
  while (parents[v] != parents[parents[v]]) {
    parents[v] = parents[parents[v]];
    v = parents[v];
  }
  return parents[v];
}

inline uintE peek_root(const sequence<parent>& parents, uintE v) {
  while (true) {
    uintE p = parents[v];
    uintE gp = parents[p];
    if (p == gp) return p;
    v = p;
  }
}

inline sequence<parent> canonicalize_labels(const sequence<parent>& labels) {
  std::unordered_map<parent, parent> remap;
  sequence<parent> result(labels.size());
  parent next = 0;
  for (size_t i = 0; i < labels.size(); ++i) {
    parent lbl = labels[i];
    auto it = remap.find(lbl);
    if (it == remap.end()) {
      it = remap.emplace(lbl, next++).first;
    }
    result[i] = it->second;
  }
  return result;
}

inline bool is_root(const sequence<parent>& parents, uintE v) {
  return parents[v] == v;
}

inline bool depth_leq_one(const sequence<parent>& parents, uintE v) {
  return parents[parents[v]] == parents[v];
}

inline sequence<Edge> rebuild_edges(sequence<parent>& parents,
                                    const sequence<Edge>& original) {
  size_t count = 0;
  for (const auto& [u, v] : original) {
    uintE ru = find_root(parents, u);
    uintE rv = find_root(parents, v);
    if (ru == rv) continue;
    if (ru > rv) std::swap(ru, rv);
    ++count;
  }

  sequence<Edge> edges(count);
  size_t idx = 0;
  for (const auto& [u, v] : original) {
    uintE ru = find_root(parents, u);
    uintE rv = find_root(parents, v);
    if (ru == rv) continue;
    if (ru > rv) std::swap(ru, rv);
    edges[idx++] = Edge{ru, rv};
  }

  auto edges_slice = parlay::make_slice(edges);
  parlay::sort_inplace(edges_slice);
  return parlay::unique(edges_slice);
}

inline sequence<parent> easy_case(size_t n, const sequence<Edge>& edges) {
  sequence<parent> P(n), Scratch(n);
  gbbs::parallel_for(0, n, [&](size_t i) { P[i] = static_cast<parent>(i); });

  const uintE kInvalid = std::numeric_limits<uintE>::max();
  sequence<uint8_t> was_root(n);
  sequence<uintE>   child_root(n);
  sequence<uint8_t> leaf_of_root(n);
  sequence<uint8_t> got_incoming(n);

  auto halve = [&](sequence<parent>& in, sequence<parent>& out) {
    gbbs::parallel_for(0, n, [&](size_t i) { out[i] = in[in[i]]; });
  };

  auto peek_root_now = [&](uintE x, const sequence<parent>& A) {
    while (true) {
      uintE p  = A[x];
      uintE gp = A[p];
      if (p == gp) return p;
      x = p;
    }
  };

  while (true) {
    halve(P, Scratch);
    std::swap(P, Scratch);

    gbbs::parallel_for(0, n, [&](size_t i) { got_incoming[i] = 0; });

    gbbs::parallel_for(0, n, [&](size_t i) {
      parent p = P[i];
      was_root[i] = (p == i);
      uintE rp = P[p];
      child_root[i] = (rp == p) ? p : kInvalid;
    });

    gbbs::parallel_for(0, edges.size(), [&](size_t ei) {
      auto [u, v] = edges[ei];
      uintE ru = child_root[u];
      uintE rv = child_root[v];
      if (ru == kInvalid || rv == kInvalid || ru == rv) return;
      uintE hi = (ru > rv) ? ru : rv;
      uintE lo = hi ^ ru ^ rv;
      gbbs::write_min<parent>(&P[hi], static_cast<parent>(lo));
    });

    gbbs::parallel_for(0, n, [&](size_t r) {
      if (was_root[r]) {
        parent to = P[r];
        if (to != r) {
          gbbs::write_max<uint8_t>(&got_incoming[to], static_cast<uint8_t>(1));
        }
      }
    });

    gbbs::parallel_for(0, n, [&](size_t i) {
      parent p = P[i];
      leaf_of_root[i] = (p != i) && (P[p] == p);
    });

    gbbs::parallel_for(0, edges.size(), [&](size_t ei) {
      auto [u, v] = edges[ei];
      bool ul = leaf_of_root[u];
      bool vl = leaf_of_root[v];
      if (ul == vl) return;

      uintE star_root = ul ? static_cast<uintE>(P[u]) : static_cast<uintE>(P[v]);
      if (got_incoming[star_root]) return;
      uintE other = ul ? v : u;

      uintE other_root = peek_root_now(other, P);
      if (other_root == star_root) return;

      gbbs::CAS<parent>(&P[star_root],
                        static_cast<parent>(star_root),
                        static_cast<parent>(other_root));
    });

    halve(P, Scratch);
    std::swap(P, Scratch);

    std::atomic<bool> any_deep(false);
    gbbs::parallel_for(0, n, [&](size_t i) {
      if (any_deep.load(std::memory_order_relaxed)) return;
      if (P[P[i]] != P[i]) any_deep.store(true, std::memory_order_relaxed);
    });
    if (any_deep.load(std::memory_order_relaxed)) continue;

    std::atomic<bool> any_live(false);
    gbbs::parallel_for(0, edges.size(), [&](size_t ei) {
      if (any_live.load(std::memory_order_relaxed)) return;
      auto [u, v] = edges[ei];
      if (P[u] != P[v]) any_live.store(true, std::memory_order_relaxed);
    });
    if (!any_live.load(std::memory_order_relaxed)) break;
  }

  return P;
}

}

template <class Graph>
sequence<parent> CC(const Graph& G, GazitParams params = GazitParams()) {
  const size_t n = G.n;

  auto edges = parlay::map(G.edges(), [](const auto& entry) {
    uintE u, v; gbbs::empty _;
    std::tie(u, v, _) = entry;
    return internal::Edge{u, v};
  });

  if (params.deduplicate_edges) {
    auto edges_slice = parlay::make_slice(edges);
    parlay::sort_inplace(edges_slice);
    edges = parlay::unique(edges_slice);
  }

  auto parents = internal::easy_case(n, edges);

  gbbs::parallel_for(0, n, [&](size_t i) {
    internal::find_root(parents, static_cast<uintE>(i));
  });

  return parents;
}

template <class Graph>
ComparisonStats BenchmarkPair(Graph& G, double beta, bool permute,
                              GazitParams params = GazitParams()) {
  ComparisonStats stats;
  std::cout << "Starting Work Efficient" << '\n';

  timer t;
  t.start();
  auto work_components =
      workefficient_cc::CC(G, beta, /*pack=*/false, /*permute=*/permute);
  stats.workefficient_time = t.stop();

  std::cout << "Starting Gazit" << '\n';

  timer gazit_timer;
  gazit_timer.start();
  auto gazit_components = CC(G, params);
  stats.gazit_time = gazit_timer.stop();

  auto work_labels = internal::canonicalize_labels(work_components);
  auto gazit_labels = internal::canonicalize_labels(gazit_components);
  if (work_labels != gazit_labels) {
    if (G.n <= 128) {
      std::cerr << "work labels:";
      for (size_t i = 0; i < G.n; ++i) std::cerr << " " << work_labels[i];
      std::cerr << "\n";
      std::cerr << "gazit labels:";
      for (size_t i = 0; i < G.n; ++i) std::cerr << " " << gazit_labels[i];
      std::cerr << "\n";
    }
    size_t mismatch_count = 0;
    size_t first_mismatch = G.n;
    for (size_t i = 0; i < G.n; ++i) {
      if (work_labels[i] != gazit_labels[i]) {
        ++mismatch_count;
        if (first_mismatch == G.n) {
          first_mismatch = i;
        }
      }
    }
    std::cerr << "Mismatch between WorkEfficient and Gazit connectivity labels" << std::endl;
    std::cerr << "  mismatching vertices: " << mismatch_count << " / " << G.n << std::endl;
    if (first_mismatch < G.n) {
      std::cerr << "  first mismatch at vertex " << first_mismatch
                << " (workefficient=" << work_labels[first_mismatch]
                << ", gazit=" << gazit_labels[first_mismatch] << ")" << std::endl;
    }
    std::ofstream diag("gazit_mismatch.txt");
    if (diag.is_open()) {
      diag << "Mismatch between WorkEfficient and Gazit connectivity labels\n";
      diag << "Total mismatching vertices: " << mismatch_count << " / " << G.n << "\n";
      diag << "work labels:";
      for (size_t i = 0; i < G.n; ++i) diag << " " << work_labels[i];
      diag << "\n";
      diag << "gazit labels:";
      for (size_t i = 0; i < G.n; ++i) diag << " " << gazit_labels[i];
      diag << "\n";
      for (size_t i = 0; i < G.n; ++i) {
        if (work_labels[i] != gazit_labels[i]) {
          diag << "v=" << i << " work=" << work_labels[i]
               << " gazit=" << gazit_labels[i] << "\n";
        }
      }
      diag.close();
    }
    abort();
  }

  work_components.clear();
  gazit_components.clear();

  return stats;
}

}
}
