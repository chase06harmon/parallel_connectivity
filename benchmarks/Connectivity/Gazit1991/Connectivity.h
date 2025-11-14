
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
#include <thread>
#include <barrier>
#include <vector>

#include "benchmarks/Connectivity/WorkEfficientSDB14/Connectivity.h"
#include "benchmarks/Connectivity/common.h"
#include "gbbs/gbbs.h"
#include "parlay/delayed_sequence.h"
#include "parlay/primitives.h"
#include "parlay/random.h"
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
  const size_t m = original.size();
  if (m == 0) {
    return sequence<Edge>();
  }

  auto root_edges = sequence<Edge>::uninitialized(m);
  auto flags = sequence<size_t>(m + 1, static_cast<size_t>(0));

  gbbs::parallel_for(0, m, [&](size_t i) {
    const auto [u, v] = original[i];
    uintE ru = find_root(parents, u);
    uintE rv = find_root(parents, v);
    if (ru == rv) {
      flags[i] = 0;
    } else {
      flags[i] = 1;
      root_edges[i] = Edge{ru, rv};
    }
  });

  size_t count = parlay::scan_inplace(parlay::make_slice(flags));
  sequence<Edge> edges(count);

  gbbs::parallel_for(0, m, [&](size_t i) {
    if (flags[i] != flags[i + 1]) {
      edges[flags[i]] = root_edges[i];
    }
  });

  return edges;
}

struct DenseToEasyResult {
  sequence<parent> parents;
  sequence<Edge> root_edges;
};

inline sequence<size_t> sample_indices(size_t m, size_t k, uint64_t seed) {
  sequence<size_t> idx(k);
  parlay::random rng(seed);
  gbbs::parallel_for(0, k, [&](size_t i) {
    auto local_rng = rng.fork(i);
    idx[i] = local_rng.ith_rand(0) % m;
  });
  return idx;
}

inline void halve_once(sequence<parent>& P) {
  size_t n = P.size();
  gbbs::parallel_for(0, n, [&](size_t i) {
    P[i] = P[P[i]];
  });
}

inline DenseToEasyResult dense_to_easy(size_t n,
                                       const sequence<Edge>& original_edges,
                                       const GazitParams& params) {
  sequence<parent> P(n);
  gbbs::parallel_for(0, n, [&](size_t i) { P[i] = static_cast<parent>(i); });

  const size_t m = original_edges.size();
  if (m == 0 || n == 0) {
    return DenseToEasyResult{std::move(P), sequence<Edge>()};
  }

  sequence<uint8_t> ext(n, static_cast<uint8_t>(1));
  sequence<uint8_t> got_incoming(n, static_cast<uint8_t>(0));

  const double alpha = (params.alpha > 1.0) ? params.alpha : 1.5;
  const size_t P_budget =
      (params.processor_budget > 0) ? params.processor_budget
                                    : std::max<size_t>(1, static_cast<size_t>(std::thread::hardware_concurrency()));
  const size_t max_rounds = (params.max_rounds > 0)
      ? params.max_rounds
      : static_cast<size_t>(std::max<double>(
            1.0,
            std::ceil(std::log(static_cast<double>(std::max<size_t>(n, 2))) / std::log(1.5))));

  for (size_t round = 0; round < max_rounds; ++round) {
    halve_once(P);

    gbbs::parallel_for(0, n, [&](size_t i) { got_incoming[i] = 0; });

    size_t sample_size = std::max<size_t>(
        P_budget,
        static_cast<size_t>(std::ceil(static_cast<double>(m) / std::pow(alpha, static_cast<double>(round))))
    );
    sample_size = std::min(sample_size, m);

    if (sample_size == 0) break;

    auto sample = sample_indices(m, sample_size, params.seed + static_cast<uint64_t>(round) * 0x9e3779b97f4a7c15ULL);

    gbbs::parallel_for(0, sample_size, [&](size_t si) {
      const auto [u0, v0] = original_edges[sample[si]];
      uintE ru = peek_root(P, u0);
      uintE rv = peek_root(P, v0);
      if (ru == rv) return;

      if ((ext[ru] ^ ext[rv]) == 0) return;

      bool u_child_of_root = depth_leq_one(P, u0);
      bool v_child_of_root = depth_leq_one(P, v0);
      if (!(u_child_of_root && v_child_of_root)) return;

      uintE r_u = P[u0];
      uintE r_v = P[v0];
      if (r_u == r_v) return;

      uintE hi = (r_u > r_v) ? r_u : r_v;
      uintE lo = hi ^ r_u ^ r_v;

      if (gbbs::CAS<parent>(&P[hi], static_cast<parent>(hi), static_cast<parent>(lo))) {
        gbbs::write_max<uint8_t>(&got_incoming[hi], static_cast<uint8_t>(1));
        gbbs::write_max<uint8_t>(&got_incoming[lo], static_cast<uint8_t>(1));
      }
    });

    gbbs::parallel_for(0, sample_size, [&](size_t si) {
      const auto [u0, v0] = original_edges[sample[si]];
      uintE ru = peek_root(P, u0);
      uintE rv = peek_root(P, v0);
      if (ru == rv) return;

      if ((ext[ru] | ext[rv]) == 0) return;

      if (depth_leq_one(P, u0)) {
        uintE r = P[u0];
        if (got_incoming[r] == 0 && r != rv) {
          uintE hi = (r > rv) ? r : rv;
          uintE lo = hi ^ r ^ rv;
          if (gbbs::CAS<parent>(&P[hi], static_cast<parent>(hi), static_cast<parent>(lo))) {
            gbbs::write_max<uint8_t>(&got_incoming[hi], static_cast<uint8_t>(1));
          }
        }
      }

      if (depth_leq_one(P, v0)) {
        uintE r = P[v0];
        if (got_incoming[r] == 0 && r != ru) {
          uintE hi = (r > ru) ? r : ru;
          uintE lo = hi ^ r ^ ru;
          if (gbbs::CAS<parent>(&P[hi], static_cast<parent>(hi), static_cast<parent>(lo))) {
            gbbs::write_max<uint8_t>(&got_incoming[hi], static_cast<uint8_t>(1));
          }
        }
      }
    });

    halve_once(P);

    gbbs::parallel_for(0, n, [&](size_t i) {
      if (P[i] == i && got_incoming[i] == 0) {
        ext[i] = 0;
      }
    });

    bool any_ext = parlay::any_of(
        parlay::delayed_seq<bool>(n, [&](size_t i) {
          return (P[i] == i) && (got_incoming[i] != 0);
        }),
        [](bool v) { return v; });
    if (!any_ext) break;
  }

  gbbs::parallel_for(0, n, [&](size_t i) { find_root(P, static_cast<uintE>(i)); });

  auto root_edges = rebuild_edges(P, original_edges);

  return DenseToEasyResult{std::move(P), std::move(root_edges)};
}

inline sequence<parent> easy_case_from(size_t n,
                                       const sequence<Edge>& edges,
                                       sequence<parent> P) {
  sequence<parent> Scratch(n);

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

    gbbs::parallel_for(0, n, [&](size_t i) {
      got_incoming[i] = 0;
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

    gbbs::parallel_for(0, n, [&](size_t i) {
      if (was_root[i]) {
        parent to = P[i];
        if (to != i) {
          gbbs::write_max<uint8_t>(&got_incoming[to], static_cast<uint8_t>(1));
        }
      }
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

    bool any_deep = parlay::any_of(
        parlay::delayed_seq<bool>(n, [&](size_t i) {
          return P[P[i]] != P[i];
        }),
        [](bool v) { return v; });
    if (any_deep) continue;

    bool any_live = parlay::any_of(
        parlay::delayed_seq<bool>(edges.size(), [&](size_t ei) {
          auto [u, v] = edges[ei];
          return P[u] != P[v];
        }),
        [](bool v) { return v; });
    if (!any_live) break;
  }

  return P;
}

inline int serial_1(uintE a, uintE b) {
  uintE diff = a ^ b;
  return __builtin_ctzll(diff);
}

inline void deterministic_mate(
  const sequence<uintE>& V_roots, // vertices currently under consideration (numbers)
  // const sequence<uint8_t>& active, // is global vertex index active in the active set
  sequence<parent>& P, // global parent array
  sequence<int>& next, // global for vertices in v (all else -1)
  sequence<int>& mate_j, // the round in which v was mated (for later use by partitioning)
  int k, // the k param
  int j, // the j round in partitioning
  int num_threads
) {
  size_t n = P.size();
  size_t n_roots = V_roots.size();

  sequence<uint8_t> removed(n, 0);
  sequence<uintE> in_deg(n, 0);

  sequence<uintE> prev(n);
  gbbs::parallel_for(0, n, [&](size_t i) {
    prev[next[i]] = i;
  });

  // compute the in-degree from the next array
  gbbs::parallel_for(0, n_roots, [&](size_t i) {
    uintE v = V_roots[i];
    gbbs::fetch_and_add(&in_deg[next[v]], 1);
  });


  gbbs::parallel_for(0, n_roots, [&](size_t i) {
    uintE v = V_roots[i];
    
    if (in_deg[v] == 0) {
      P[v] = next[v];
      mate_j[v] = j; 
      mate_j[next[v]] = j;

      removed[next[v]] = 1; // remove parents of zero degree vertex
    };

    if (in_deg[v] == 0 || in_deg[v] >= 2) {
      removed[v] = 1; // remove in_deg 0, 2+
    }
  });

  // how to deal with cycles?
  // for (int i = 0; i < log2(n_roots); i++) {
  //   gbbs::parallel_for(0, n_roots, [&](size_t i) {
  //     if (removed[i] || removed[next[i]]) return;
  //     next[i] = next[next[i]];
  //   });
  // }

  // gbbs::parallel_for(0, n_roots, [&](size_t i) {
  //   if (removed[i]) return;

  //   P[i] = next[i];
  // });

  // NOTE: From here on out must use removed to make sure v hasn't been removed from the graph


  // Can't do pointer jumping (not work efficient)

  // NOTE: pointer jumping probably faster... 

  std::barrier sync_round(num_threads);
  std::barrier sync_chosen(num_threads);

  std::vector<int> round_taken(n, -1);
  size_t chunk_size = (n_roots + num_threads - 1) / num_threads; 

  auto worker = [&](int tid) {
    int lo = tid * chunk_size; 
    int hi = std::min(n_roots, lo+chunk_size);

    int cur_pos = lo;
    int round = 0;

    while (round < chunk_size + 2*log2(log2(n))) {
      if (cur_pos >= hi) { // no more vertices to process
        sync_chosen.arrive_and_wait();
        sync_round.arrive_and_wait();
        round++;
        continue;
      }

      if (removed[V_roots[cur_pos]]) { // if the vertex has been removed, we don't need to consider it
        cur_pos++;
        continue;
      }

      uintE v = V_roots[cur_pos];
      round_taken[v] = round;

      // if (tid == 1) {
      //   std::cout << "v: " << v << std::endl;
      // }

      sync_chosen.arrive_and_wait();

      if (round_taken[next[v]] == round) { // SERIAL calculation and work backwards
        uintE next_v = next[v]; 
        uintE prev_v = prev[v];
        uint8_t is_tail = 0; 

        if (next[next_v] < 0 || round_taken[next[next_v]] != round) {
          is_tail = 1;

        } else if (prev_v < 0 || round_taken[prev_v] != round) {
          is_tail = 1;

        } else {
          int my_val = serial_1(v, next_v);
          int prev_val = serial_1(prev_v, v);
          int next_val = serial_1(next_v, next[next_v]);

          if ((my_val > prev_val) || (my_val == prev_val && (v & (1u << my_val))) && (my_val > next_val) || (my_val == next_val && (v & (1u << my_val)))) {
            is_tail = 1;
          } else {
            is_tail = 0;
          }
        }

        if (is_tail) {
          int freeze_round = round;

          sync_round.arrive_and_wait();
          round++;
          
          uintE cur_v = v;
          
          while (1) {
            cur_v = prev[cur_v];

            uintE next_v = next[cur_v]; 
            uintE prev_v = prev[cur_v];
            uint8_t is_tail = 0;

            if (prev_v < 0 || round_taken[prev_v] != freeze_round) {
              break;
            }

            int my_val = serial_1(cur_v, next_v);
            int prev_val = serial_1(prev_v, cur_v);
            int next_val = serial_1(next_v, next[next_v]);

            if ((my_val > prev_val) || (my_val == prev_val && (cur_v & (1u << my_val))) && (my_val > next_val) || (my_val == next_val && (cur_v & (1u << my_val)))) {
              break;
            }

            sync_chosen.arrive_and_wait();

            P[cur_v] = v;
            sync_round.arrive_and_wait();
            round++;
          }
        } else {
          sync_round.arrive_and_wait();
          round++;
        }

      } else {
        P[v] = P[next[v]];
        removed[next[v]] = 1;
        cur_pos++;
        sync_round.arrive_and_wait();
        round++;
      }
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  for (int t = 0; t < num_threads; ++t) {
      threads.emplace_back(worker, t);
  }
  for (auto& th : threads) th.join();

}

inline std::pair<sequence<uintE>, sequence<uintE>> partitioning(
  size_t n,
  sequence<uintE>& V, 
  sequence<Edge>& E
) {
  // double alpha = 0.5;

  // sequence<uintE> P(n);
  // sequence<uint32_t> flag(n, 0);
  // sequence

  // gbbs::parallel_for(0, n, [&](size_t i) {P[i] = static_cast<parent>(i);});

  // for (int j = 0; j < ceil(2*log(log(n))); j++) {
  //   double p  = E.size() * pow(alpha, j);
  //   uint64_t seed = j;

  //   auto keep = gbbs::filter(gbbs::iota(E.size()))
  // }
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

    gbbs::parallel_for(0, n, [&](size_t i) {
      got_incoming[i] = 0;
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

    gbbs::parallel_for(0, n, [&](size_t i) {
      if (was_root[i]) {
        parent to = P[i];
        if (to != i) {
          gbbs::write_max<uint8_t>(&got_incoming[to], static_cast<uint8_t>(1));
        }
      }
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

    bool any_deep = parlay::any_of(
        parlay::delayed_seq<bool>(n, [&](size_t i) {
          return P[P[i]] != P[i];
        }),
        [](bool v) { return v; });
    if (any_deep) continue;

    bool any_live = parlay::any_of(
        parlay::delayed_seq<bool>(edges.size(), [&](size_t ei) {
          auto [u, v] = edges[ei];
          return P[u] != P[v];
        }),
        [](bool v) { return v; });
    if (!any_live) break;
  }

  return P;
}

/*
Summary of High-Level Steps

Start with a sparse graph.

Iteratively partition it into dense and sparse subsets.

Sample edges to find dense (“extrovert”) vertices.

Merge extroverts into supervertices via deterministic mating.

Replace edges to connect only supervertex roots.

Repeat until graph size ≤ 
𝑛
/
log
⁡
𝑛
n/logn.

Proceed with dense-to-easy reduction and then the easy-case algorithm.
*/

}

void test_deterministic_mate() {
  size_t N = 100000;

  sequence<uintE> V_roots(N);
  sequence<uintE> P(N);
  sequence<int> next(N, -1);
  sequence<int> mate_j(N, -1);
  int k = 0; 
  int j = 0; 
  int num_threads = N / log2(N);

  gbbs::parallel_for(0, N, [&](size_t i){V_roots[i] = i; P[i] = i; next[i] = (i+1) % N;});


  // size_t N = 9;

  // sequence<uintE> V_roots(N);
  // sequence<uintE> P(N);
  // sequence<int> mate_j(N, -1);
  // int k = 0; 
  // int j = 0; 
  // int num_threads = N / log2(N);

  // gbbs::parallel_for(0, N, [&](size_t i){V_roots[i] = i; P[i] = i;});

  // sequence<int> next = {1, 2, 1, 4, 2, 3, 7, 8, 4};

  internal::deterministic_mate(V_roots, P, next, mate_j, 0, 0, num_threads);

  std::cout << "Num threads: " << num_threads << std::endl;
  std::cout << "Per thread: " << (N + num_threads - 1) / num_threads  << std::endl;

  // for (int i = 0; i < N; i++) {
  //   // if (P[i] == P[i+1]) {
  //     std::cout << "i: " << i << " p: " << P[i] << std::endl;
  //   // }
  // }

}

template <class Graph>
sequence<parent> CC(const Graph& G, GazitParams params = GazitParams()) {
  const size_t n = G.n;

  auto edges = parlay::map(G.edges(), [](const auto& entry) {
    uintE u, v; gbbs::empty _;
    std::tie(u, v, _) = entry;
    return internal::Edge{u, v};
  });

  std::cout << "[gazit] edges before dense_to_easy: "
            << edges.size() << std::endl;
  auto de = internal::dense_to_easy(n, edges, params);
  std::cout << "[gazit] edges after dense_to_easy: "
            << de.root_edges.size() << std::endl;

  if (de.root_edges.size() == 0) {
    gbbs::parallel_for(0, n, [&](size_t i) {
      internal::find_root(de.parents, static_cast<uintE>(i));
    });
    return de.parents;
  }

  auto parents = internal::easy_case_from(n, de.root_edges, std::move(de.parents));

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
