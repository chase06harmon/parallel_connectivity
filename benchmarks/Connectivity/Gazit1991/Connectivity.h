
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
#include "parlay/utilities.h"

namespace gbbs {
namespace gazit_cc {

struct GazitParams {
  double alpha = 1.5;
  size_t processor_budget = 0;
  size_t max_rounds = 1;
  uint64_t seed = 5489;
  bool skip_sparse_to_dense = false;
  bool easy_case_only = false;
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

inline bool child_of_root(const sequence<parent>& P, uintE x) {
  parent p = P[x];
  return P[p] == p;
}

inline uint8_t extparent(const sequence<parent>& P,
                         const sequence<uint8_t>& ext,
                         uintE x) {
  return ext[P[x]];
}

inline void set_flag1(uint8_t* addr) {
  gbbs::write_max<uint8_t>(addr, static_cast<uint8_t>(1));
}
inline uint8_t load_flag(uint8_t* addr) {
  return gbbs::atomic_load(addr);
}
inline uint8_t load_u8(const uint8_t* addr) {
  return gbbs::atomic_load<uint8_t>(const_cast<uint8_t*>(addr));
}
inline void store_u8(uint8_t* addr, uint8_t v) {
  gbbs::atomic_store<uint8_t>(addr, v);
}

// Mark with epoch, but avoid redundant stores once already marked this epoch.
inline void mark_epoch(uint8_t* addr, uint8_t epoch) {
  uint8_t cur = load_u8(addr);
  if (cur != epoch) store_u8(addr, epoch);
}
inline bool is_marked_epoch(const uint8_t* addr, uint8_t epoch) {
  return load_u8(addr) == epoch;
}

// Advance epoch; handle wrap by clearing the array with atomic_store(0).
inline uint8_t advance_epoch(sequence<uint8_t>& changed, uint8_t epoch) {
  epoch = static_cast<uint8_t>(epoch + 1);
  if (epoch == 0) {  // wrapped
    gbbs::parallel_for(0, changed.size(), [&](size_t i) {
      store_u8(&changed[i], static_cast<uint8_t>(0));
    });
    epoch = 1;
  }
  return epoch;
}
inline sequence<parent> canonicalize_labels(const sequence<parent>& labels) {
  const size_t n = labels.size();
  if (n == 0) return sequence<parent>();

  // Domain is 0..n-1 for CC roots; if not guaranteed, compute max label first.
  const size_t dom = n;

  // first_pos[l] = earliest index where label l appears; n means absent.
  sequence<size_t> first_pos(dom, n);
  gbbs::parallel_for(0, n, [&](size_t i) {
    parent l = labels[i];
    gbbs::write_min<size_t>(&first_pos[l], i);  // atomic min
  });

  // Collect present labels.
  auto present = parlay::filter(parlay::iota<parent>(dom),
                                [&](parent l){ return first_pos[l] != n; });

  // Sort present labels by their first appearance (radix/integer sort is ideal here).
  parlay::integer_sort_inplace(present, [&](parent l){ return (uint64_t)first_pos[l]; });

  // id_for_label[l] = rank of l by first appearance.
  sequence<parent> id_for_label(dom, std::numeric_limits<parent>::max());
  gbbs::parallel_for(0, present.size(), [&](size_t i){
    id_for_label[present[i]] = (parent)i;
  });

  // Produce output mapping.
  sequence<parent> out(n);
  gbbs::parallel_for(0, n, [&](size_t i){
    out[i] = id_for_label[labels[i]];
  });
  return out;
}


inline bool is_root(const sequence<parent>& parents, uintE v) {
  return parents[v] == v;
}

inline bool depth_leq_one(const sequence<parent>& parents, uintE v) {
  return parents[parents[v]] == parents[v];
}

// NOTE: Seems a bit weird
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

inline void halve_once(sequence<parent>& P) {
  size_t n = P.size();
  gbbs::parallel_for(0, n, [&](size_t i) {
    P[i] = P[P[i]];
  });
}

inline void try_star_side_epoch(uintE other,
                                parent r,  // root(from) (i.e., P[from] where from is child-of-root)
                                sequence<parent>& P,
                                sequence<uint8_t>& changed,
                                uint8_t epoch) {
  if (is_marked_epoch(&changed[r], epoch)) return;

  parent pother = P[other];           // parent(other), exactly once
  bool other_child = (P[pother] == pother);

  // If other is child-of-root, live iff parents differ.
  if (other_child && pother == r) return;

  parent to = pother; // parent(other) (NOT root(other))
  if (gbbs::CAS<parent>(&P[r], r, to)) {
    // Receiver gets a new incoming edge only when receiver is a root (other_child).
    if (other_child) mark_epoch(&changed[to], epoch);
  }
}

inline DenseToEasyResult dense_to_easy(size_t n,
                                       const sequence<Edge>& original_edges,
                                       const GazitParams& params,
                                      sequence<parent> P) {

  const size_t m = original_edges.size();
  if (m == 0 || n == 0) {
    return DenseToEasyResult{std::move(P), sequence<Edge>()};
  }

  sequence<uint8_t> ext(n, static_cast<uint8_t>(1));
  sequence<uint8_t> changed(n, static_cast<uint8_t>(0));
  sequence<parent> Scratch(n);
  sequence<size_t> sample;

  const double alpha = std::max(params.alpha, 1.0);
  const size_t P_budget =
      (params.processor_budget > 0) ? params.processor_budget
                                    : std::max<size_t>(1, parlay::num_workers());
  const size_t max_rounds = params.max_rounds;
  double cur = static_cast<double>(m);

  auto any_ext = [&](const sequence<parent>& Parents,
                     const sequence<uint8_t>& Ext) {
    return parlay::any_of(
        parlay::delayed_seq<bool>(n, [&](size_t i) {
          return (Parents[i] == i) && (Ext[i] != 0);
        }),
        [](bool v) { return v; });
  };

  size_t round = 0;
  uint8_t epoch = 0;
  while (true) {
    if (max_rounds > 0 && round >= max_rounds) break;

    epoch = advance_epoch(changed, epoch);

    // Step 1: jump-over and mark non-star roots (epoch-stamped via atomic helpers; no per-iter clear).
    gbbs::parallel_for(0, n, [&](size_t i) {
      parent p = P[i];
      parent gp = P[p];
      if (p != gp && P[gp] == gp) {
        mark_epoch(&changed[gp], epoch);
      }
      Scratch[i] = gp;
    });
    P.swap(Scratch);

    size_t sample_size = std::max<size_t>(
        P_budget,
        static_cast<size_t>(std::ceil(cur)));
    sample_size = std::min(sample_size, m);
    cur /= alpha;
    if (sample_size == 0) break;

    bool sample_all = sample_size == m;
    if (!sample_all) {
      sample.resize(sample_size);
      parlay::random_generator gen(params.seed + static_cast<uint64_t>(round) * 0x9e3779b97f4a7c15ULL);
      gbbs::parallel_for(0, sample_size, [&](size_t i) {
        sample[i] = static_cast<size_t>(gen[i]() % m);
      });
    }

    auto process_step2 = [&](const Edge& e) {
      uintE u = e.first, v = e.second;

      parent pu = P[u];
      parent pv = P[v];

      // extparent guard (both endpoints extrovert)
      if (ext[pu] == 0 || ext[pv] == 0) return;

      // both endpoints must be children-of-roots
      if (P[pu] != pu || P[pv] != pv) return;

      if (pu == pv) return;

      parent hi = (pu > pv) ? pu : pv;
      parent lo = (hi == pu) ? pv : pu;

      if (gbbs::CAS<parent>(&P[hi], hi, lo)) {
        mark_epoch(&changed[lo], epoch);
      }
    };

    // Step 2: root-hook sampled edges (minimized P[] loads; changed uses epoch+atomic helpers).
    if (sample_all) {
      gbbs::parallel_for(0, m, [&](size_t i) { process_step2(original_edges[i]); });
    } else {
      gbbs::parallel_for(0, sample_size, [&](size_t i) { process_step2(original_edges[sample[i]]); });
    }

    auto process_step3 = [&](const Edge& e) {
      uintE u = e.first, v = e.second;

      parent pu = P[u];
      parent pv = P[v];

      if (ext[pu] == 0 || ext[pv] == 0) return;

      // Step 3: star-hook with minimized loads and epoch-stamped (atomic) changed flags.
      if (P[pu] == pu) {
        try_star_side_epoch(v, pu, P, changed, epoch);
      }
      if (P[pv] == pv) {
        try_star_side_epoch(u, pv, P, changed, epoch);
      }
    };

    if (sample_all) {
      gbbs::parallel_for(0, m, [&](size_t i) { process_step3(original_edges[i]); });
    } else {
      gbbs::parallel_for(0, sample_size, [&](size_t i) { process_step3(original_edges[sample[i]]); });
    }

    // Step 4: jump-over and retire roots that saw no changes.
    gbbs::parallel_for(0, n, [&](size_t i) {
      Scratch[i] = P[P[i]];
    });
    P.swap(Scratch);

    gbbs::parallel_for(0, n, [&](size_t i) {
      if (P[i] == i && !is_marked_epoch(&changed[i], epoch)) {
        ext[i] = static_cast<uint8_t>(0);
      }
    });

    ++round;
    if (!any_ext(P, ext)) break;
  }

  gbbs::parallel_for(0, n, [&](size_t i) { find_root(P, static_cast<uintE>(i)); });

  sequence<Edge> root_edges;
  {
    auto mapped = sequence<Edge>::uninitialized(m);
    auto flags = sequence<size_t>(m + 1, 0);

    gbbs::parallel_for(0, m, [&](size_t i) {
      auto [u, v] = original_edges[i];
      parent ru = P[u];
      parent rv = P[v];
      if (ru == rv) {
        flags[i] = 0;
      } else {
        flags[i] = 1;
        mapped[i] = Edge{static_cast<uintE>(ru), static_cast<uintE>(rv)};
      }
    });

    size_t count = parlay::scan_inplace(parlay::make_slice(flags));
    sequence<Edge> edges(count);

    gbbs::parallel_for(0, m, [&](size_t i) {
      if (flags[i] != flags[i + 1]) {
        edges[flags[i]] = mapped[i];
      }
    });

    root_edges = std::move(edges);
  }

  return DenseToEasyResult{std::move(P), std::move(root_edges)};
}

inline sequence<parent> easy_case_from(size_t n,
                                       const sequence<Edge>& edges,
                                       sequence<parent> P) {
  sequence<parent> Scratch(n);
  sequence<uint8_t> changed(n, static_cast<uint8_t>(0));
  const size_t m = edges.size();
  uint8_t epoch = 0;

  while (true) {
    epoch = advance_epoch(changed, epoch);

    // Step 1: jump-over and mark non-star roots (epoch-stamped via atomic helpers; no per-iter clear).
    gbbs::parallel_for(0, n, [&](size_t i) {
      parent p = P[i];
      parent gp = P[p];
      if (p != gp && P[gp] == gp) {
        mark_epoch(&changed[gp], epoch);
      }
      Scratch[i] = gp;
    });
    P.swap(Scratch);

    // Step 2: root-hook on all edges (minimized P[] loads; epoch-stamped marks).
    gbbs::parallel_for(0, m, [&](size_t ei) {
      uintE u = edges[ei].first, v = edges[ei].second;

      parent pu = P[u];
      parent pv = P[v];

      // both endpoints must be children-of-roots
      if (P[pu] != pu || P[pv] != pv) return;

      if (pu == pv) return;

      parent hi = (pu > pv) ? pu : pv;
      parent lo = (hi == pu) ? pv : pu;
      if (gbbs::CAS<parent>(&P[hi], hi, lo)) {
        mark_epoch(&changed[lo], epoch);
      }
    });

    // Step 3: star-hook on all edges with minimized loads and epoch-stamped (atomic) flags.
    gbbs::parallel_for(0, m, [&](size_t ei) {
      uintE u = edges[ei].first, v = edges[ei].second;

      parent pu = P[u];
      parent pv = P[v];

      if (P[pu] == pu) {
        try_star_side_epoch(v, pu, P, changed, epoch);
      }
      if (P[pv] == pv) {
        try_star_side_epoch(u, pv, P, changed, epoch);
      }
    });

    // Step 4: jump-over.
    gbbs::parallel_for(0, n, [&](size_t i) {
      Scratch[i] = P[P[i]];
    });
    P.swap(Scratch);

    bool any_deep = parlay::any_of(
        parlay::delayed_seq<bool>(n, [&](size_t i) {
          return P[P[i]] != P[i];
        }),
        [](bool v) { return v; });
    if (any_deep) continue;

    bool any_live = parlay::any_of(
        parlay::delayed_seq<bool>(m, [&](size_t ei) {
          const auto [u, v] = edges[ei];
          return P[u] != P[v];
        }),
        [](bool v) { return v; });
    if (!any_live) break;
  }

  return P;
}

inline sequence<parent> easy_case(size_t n, const sequence<Edge>& edges) {
  sequence<parent> P(n);
  gbbs::parallel_for(0, n, [&](size_t i) { P[i] = static_cast<parent>(i); });
  return easy_case_from(n, edges, std::move(P));
}

inline int serial_1(uintE a, uintE b) {
  uintE diff = a ^ b;
  return __builtin_ctzll(diff);
}

struct MateScratch {
  sequence<uint8_t> removed;
  sequence<uintE>   in_deg;
  sequence<uintE>   prev;
  sequence<uint8_t> end_compression;
  sequence<int>     next_new;

  void resize_if_needed(size_t n) {
    if (removed.size() >= n) return;
      removed         = sequence<uint8_t>(n);
      in_deg          = sequence<uintE>(n);
      prev            = sequence<uintE>(n);
      end_compression = sequence<uint8_t>(n);
      next_new        = sequence<int>(n);
    }
};

inline void deterministic_mate2(
  const sequence<uintE>& V_roots, // vertices currently under consideration (numbers)
  sequence<parent>& P, // global parent array
  sequence<int>& next, // global for vertices in v (all else -1)
  sequence<int>& mate_j, // the round in which v was mated (for later use by partitioning)
  int j, // the j round in partitioning
  MateScratch& scratch
  
) {
  size_t n = P.size();
  size_t n_roots = V_roots.size();

  auto& removed         = scratch.removed;
  auto& in_deg          = scratch.in_deg;
  auto& prev            = scratch.prev;
  auto& end_compression = scratch.end_compression;
  auto& next_new        = scratch.next_new;

   gbbs::parallel_for(0, n_roots, [&](size_t i) {
      uintE v = V_roots[i];
      removed[v] = 0;
      in_deg[v]  = 0;
      prev[v]    = uintE(-1);
      end_compression[v] = 0;
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

      removed[next[v]] = 1; // remove parents of zero degree vertex
    };

    if (in_deg[v] == 0 || in_deg[v] >= 2) {
      removed[v] = 1; // remove in_deg 0, 2+
    }
  });

  gbbs::parallel_for(0, n_roots, [&](size_t i) {
    int v = V_roots[i];
    if (!removed[next[v]])
      prev[next[v]] = v;
  });


  gbbs::parallel_for(0, n_roots, [&](size_t i) {
    uintE v = V_roots[i];

    int next_v = next[v];
    int prev_v = prev[v];
    uint8_t is_tail = 0;

    if (removed[next_v]) { // if the next vertex has been removed, we should stop star compression here (cannot proceed) Note: next_v always defined
      is_tail = 1;

    } else if (prev_v < 0 || removed[prev_v]) {
      is_tail = 0;

    } else {
      int my_val = serial_1(v, next_v);
      int prev_val = serial_1(prev_v, v);
      int next_val = next[next_v] > -1 ? serial_1(next_v, next[next_v]) : -1; // if next[next_v] is not defined we cannot compute serial and thus trivially passes

      if (((my_val > prev_val) || (my_val == prev_val && (v & (1u << my_val)))) && ((my_val > next_val) || (my_val == next_val && (v & (1u << my_val))))) {
        is_tail = 1;
      } else {
        is_tail = 0;
      }
    }

    end_compression[v] = is_tail;
  });

  for (int r = 0; r < (int)std::ceil(std::log2(std::log2((double)n_roots))); r++) {
    gbbs::parallel_for(0, n_roots, [&](size_t i) {
      uintE v = V_roots[i];
      if (removed[v] || removed[next[v]]) {
        next_new[v] = next[v];
        return;
      }

      if (end_compression[v] || end_compression[next[v]]) {
        next_new[v] = next[v];
        return;
      }

      next_new[v] = next[next[v]];  // both reads are from old buffer
    });

    std::swap(next, next_new);
  }

  gbbs::parallel_for(0, n_roots, [&](size_t i) {
    uintE v = V_roots[i];

    if (removed[v]) return;

    if (end_compression[v] != 1) { // if we are the head of a sub list, our parent pointer stays the same (root of star)
      P[v] = next[v];
      mate_j[v] = j;
    }
  });

}

inline parent get_root(
  uintE v,
  sequence<parent>& P
) {
  while (P[v] != v) {
    v = P[v];
  }

  return v;
}

inline sequence<size_t> sample_indices(size_t m, size_t k, uint64_t seed) {
  sequence<size_t> idx(k);
  parlay::random rng(seed);
  gbbs::parallel_for(0, k, [&](size_t i) {
    auto local_rng = rng.fork(i);
    idx[i] = local_rng.ith_rand(0) % m;
  });
  return idx;
}


inline sequence<Edge> sample_edges(
  sequence<Edge>& E,
  size_t target_size,
  uint64_t seed = 42
) {
  size_t m = E.size();
  if (target_size >= m) {
    // Either return E by value or make a copy; choice depends on callsites.
    return E;  // if you’re fine with aliasing
  }

  auto idx = sample_indices(m, target_size, seed);
  sequence<Edge> out(target_size);

  gbbs::parallel_for(0, target_size, [&](size_t i) {
    out[i] = E[idx[i]];
  });

  return out;
}


// Note:: Does not take out [x,x] edges
inline void reassign_edges(sequence<parent>& parents, sequence<Edge>& E) {
  const size_t m = E.size();

  // std::cout << "TESTER TESTER1" << std::endl;
  // for (auto & [u,v] : E) {
  //   if (u == 0 || v == 0) {
  //     std::cout << u << ", " << v << std::endl;
  //     break;
  //   }
  // }

  // std::cout << "TESTER P[0]: " << parents[0] << std::endl;

  gbbs::parallel_for(0, m, [&](size_t i) {
    const auto [u, v] = E[i];
    if (parents[u] == static_cast<parent>(-1) || parents[v] == static_cast<parent>(-1))
      return;
    uintE ru = get_root(u, parents);
    uintE rv = get_root(v, parents);
    E[i].first = ru;
    E[i].second = rv;
  });

  // std::cout << "PAST IT" << std::endl;

  // parlay::integer_sort_inplace(
  //   E,
  //   [&](const Edge& e) -> uint64_t {
  //     return (uint64_t(e.first) << 32) | uint64_t(e.second);
  //   }
  // );

  // E = parlay::unique(E);
  E = parlay::remove_duplicates(E);
  E = parlay::filter(E, [&](Edge e) {return e.first != e.second;});
}

// Note:: Does not take out [x,x] edges
inline void reassign_edges_to_parent(sequence<parent>& parents, sequence<Edge>& E) {
  const size_t m = E.size();

  // std::cout << "TESTER TESTER1" << std::endl;
  // for (auto & [u,v] : E) {
  //   if (u == 0 || v == 0) {
  //     std::cout << u << ", " << v << std::endl;
  //     break;
  //   }
  // }

  // std::cout << "TESTER P[0]: " << parents[0] << std::endl;

  gbbs::parallel_for(0, m, [&](size_t i) {
    const auto [u, v] = E[i];
    if (parents[u] == static_cast<parent>(-1) || parents[v] == static_cast<parent>(-1))
      return;
    E[i].first = parents[u];
    E[i].second = parents[v];
  });

  E = parlay::remove_duplicates(E);
  E = parlay::filter(E, [&](Edge e) {return e.first != e.second;});
}

struct PartitioningResult {
  sequence<int> extrovert_flag;
};

inline PartitioningResult gazit_partitioning(
  sequence<uintE>& V,
  sequence<Edge>& E,
  sequence<Edge>& E_graph,
  int big_n,
  sequence<parent>& global_p
) {
  double alpha = 0.5;
  int n = V.size();
  int rounds = ceil(log2(log2(big_n)));

  // NOTE: Should be fine to init this way because edges are only between vertices in V
  // Alternative: Remap vertices and edges in wrapper
  sequence<parent> P(big_n, -1);
  sequence<int> mate_j(big_n,-1);
  sequence<int> flag(big_n, 0);

  MateScratch scratch;
  scratch.resize_if_needed(big_n);


  gbbs::parallel_for(0, n, [&](size_t i) {P[V[i]] = V[i];});

  for (int j = 0; j <= rounds; j++) {
    size_t target_size = static_cast<size_t>(E.size() * pow(alpha, j));

    sequence<Edge> E_sample = internal::sample_edges(E, target_size, 1337 + j);
    sequence<int> next(big_n, -1);

    gbbs::parallel_for(0, E_sample.size(), [&](size_t i) {
      auto [u,v] = E_sample[i];

      uintE root_u = get_root(u, P);
      uintE root_v = get_root(v, P);

      if (root_u != root_v && flag[root_u] == j && flag[root_v] == j) { // live edge
        next[root_u] = root_v;
        next[root_v] = root_u;
        flag[root_u] = j + 1;
        flag[root_v] = j + 1;
      }
    });

    sequence<uintE> V_roots = parlay::filter(V, [&](uintE v) {return flag[v] == j+1;});

    deterministic_mate2(V_roots, P, next, mate_j, j, scratch);
  }

  // int v_test = 0;

  // while (P[v_test] != -1 && v_test != P[v_test]) {
  //   std::cout << "partioning p of " << v_test << "= " << P[v_test] << std::endl;
  //   std::cout << "mate j of " << v_test << "= " << mate_j[v_test] << std::endl;
  //   v_test = P[v_test];
  // }
  // std::cout << "paritioning p[0]: " << P[0] << std::endl;
  // std::cout << "paritioning p[23977]: " << P[23977] << std::endl;
  // std::cout << "paritioning p[23129]: " << P[23129] << std::endl;
  // std::cout << "paritioning p[13573]: " << P[13573] << std::endl;
  // std::cout << "paritioning p[8973]: " << P[8973] << std::endl;

  for (int j = rounds; j >= 0; j--) {
    gbbs::parallel_for(0, n, [&](size_t i) {
      uintE v = V[i];
      if (mate_j[v] == j) {
        P[v] = P[P[v]];
      }
    });
  }

  // v_test = 0;

  // while (P[v_test] != -1 && v_test != P[v_test]) {
  //   std::cout << "partioning2 p of " << v_test << "= " << P[v_test] << std::endl;
  //   v_test = P[v_test];
  // }

  // reassign_edges(P, E_graph);
  reassign_edges_to_parent(P, E);

  sequence<int> extrovert_flag = parlay::map(flag, [&](int x){ return x == rounds + 1 ? 1 : 0;});

  // sequence<int> extrovert_flag(big_n, 0);

  // gbbs::parallel_for(0, big_n, [&](size_t i) {
  //   if (P[i] != -1 && get_root(i,P) != i) {
  //     extrovert_flag[get_root(i, P)] = 1;
  //   }
  // });

  gbbs::parallel_for(0, n, [&](size_t i){
    uintE v = V[i];
    parent p = P[v];
    global_p[v] = p;
  });

  return PartitioningResult{
    std::move(extrovert_flag),
  };
}

template<class Graph>
std::pair<sequence<uintE>, sequence<Edge>> sparse_to_dense(Graph & G, sequence<parent>& P) {
  // int m = G.m;
  int n = G.n;

  int rounds = ceil(2*log2(log2(n)));
  sequence<uintE> extrovert_set(n, 0);

  auto E = parlay::map(G.edges(), [](const auto& entry) {
    uintE u, v; gbbs::empty _;
    std::tie(u, v, _) = entry;
    return internal::Edge{u, v};
  });

  sequence<uintE> V = sequence<uintE>(G.n);
  gbbs::parallel_for(0, G.n, [&](size_t i){V[i] = i;});

  auto V_i = V;
  auto E_i = E;
  auto E_orig = E;
  // sequence<Edge> E_cum;

  // for (auto & [u,v] : E) {
  //   if (u == 0 || v == 0) {
  //     std::cout << u << ", " << v << std::endl;
  //     break;
  //   }
  // }

  // for (auto & [u,v] : E_i) {
  //   if (u == 0 || v == 0) {
  //     std::cout << u << ", " << v << std::endl;
  //     break;
  //   }
  // }


  sequence<int> extrovert_flag;

  for (int i = 0; i <= rounds; i++) { // change to rounds
  //   for (auto & [u,v] : E_i) {
  //     if (u == 0 || v == 0) {
  //       std::cout << u << ", " << v << std::endl;
  //       break;
  //     }
  //   }
    // std::cout << "V[0] r" << i << ": " << V_i[0] << std::endl;
    // std::cout << "extro[0] r" << i << ": " << extrovert_set[0] << std::endl;
    // std::cout << "extro[23977] r" << i << ": " << extrovert_set[23977] << std::endl;
    PartitioningResult partition_result = gazit_partitioning(V_i, E_i, E, n, P);
    // for (auto & [u,v] : E_i) {
    //   if (u == 0 || v == 0) {
    //     std::cout << u << ", " << v << std::endl;
    //     break;
    //   }
    // }
    extrovert_flag = std::move(partition_result.extrovert_flag);

    gbbs::parallel_for(0, V_i.size(), [&](size_t i) {
      uintE v = V_i[i];
      if (extrovert_flag[v])
        extrovert_set[v] = 1;
    });

    if (i <= rounds -1) {
      sequence<uint8_t> introvert_not_isolated(n, 0);
      sequence<uint8_t> include_in_E_i(E_i.size(), 0);

      gbbs::parallel_for(0, E_i.size(), [&](size_t i){
        auto [u,v] = E_i[i];
        if (extrovert_set[u] || extrovert_set[v]) return; // this edge is not live or one of its endpoints is extrovert

        // There is an edge from u to v and they are both introverted!
        introvert_not_isolated[u] = 1;
        introvert_not_isolated[v] = 1;
        include_in_E_i[i] = 1;
      });

      E_i = parlay::pack(E_i, include_in_E_i);
      V_i = parlay::pack(V, introvert_not_isolated);
    }

  }

  // auto introvert = parlay.pack(V, parlay::tabulate(V.size(), [&](size_t i) {
  //   return ! extrovert_set[i];
  // }));

  gbbs::parallel_for(0, E.size(), [&](size_t i){
    auto [u, v] = E[i];
    if (P[v] == v && extrovert_set[u] && !extrovert_set[v]) {
      P[v] = u;
    } else if (P[u] == u && extrovert_set[v] && !extrovert_set[u]) {
      P[u] = v;
    }
  });

  gbbs::parallel_for(0, n, [&](size_t i){
    P[i] = get_root(i, P);

    if (!extrovert_set[i] && P[i] == i) { // could be introvert roots that are isolated since they mated at a lower round. excluded from introvert AND extrovert! 
      extrovert_set[i] = 1;
    }
  });
  // std::cout << "P[0] final: " << P[0] << std::endl; // --

  // std::cout << "extro[0]" << extrovert_set[0] << std::endl;


  reassign_edges_to_parent(P, E);

  gbbs::parallel_for(0, V_i.size(), [&](size_t i) {
    uintE v = V_i[i];

    extrovert_set[v] = 1;
  });

  V = parlay::pack(V, extrovert_set);

  return std::pair{V, E};

      // Deterministic hooking below
    // gbbs::parallel_for(0, extrovert_flag.size(), [&](size_t i) {
    //   if (extrovert_flag[i]) return;

    //   uintE introvert_v = V_i[i];
    //   auto v = G.get_vertex(introvert_v);

    //   for (int r = 0; r < v.out_degree(); r++) {
    //     // gbbs::empty _;
    //     auto [n, _] = v.out_neighbors().get_ith_neighbor(r);
    //     if (extrovert_set[n]) {
    //       P[introvert_v] = n;
    //       break;
    //     }
    //   }
    // });
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

// void map_edges(
//   sequence<int>& map,
//   sequence<parent>& P
// ) {
//   gbbs::parallel_for()
// }


template <class Graph>
sequence<parent> CC(const Graph& G, GazitParams params = GazitParams()) {
  const size_t n = G.n;

  if (params.easy_case_only) {
    std::cerr << "[gazit] skipping sparse_to_dense; using original graph"
              << std::endl;

    auto edges = parlay::map(G.edges(), [](const auto& entry) {
      uintE u, v; gbbs::empty _;
      std::tie(u, v, _) = entry;
      return internal::Edge{u, v};
    });

    sequence<parent> P(n);
    gbbs::parallel_for(0, n, [&](size_t i) {
      P[i] = static_cast<parent>(i);
    });

    auto parents =
        internal::easy_case(n, edges, P);

    gbbs::parallel_for(0, n, [&](size_t i) {
      internal::find_root(parents, static_cast<uintE>(i));
    });

    return parents;
  } 

  if (params.skip_sparse_to_dense) {
    std::cerr << "[gazit] skipping sparse_to_dense; using original graph"
              << std::endl;

    auto edges = parlay::map(G.edges(), [](const auto& entry) {
      uintE u, v; gbbs::empty _;
      std::tie(u, v, _) = entry;
      return internal::Edge{u, v};
    });

    sequence<parent> P(n);
    gbbs::parallel_for(0, n, [&](size_t i) {
      P[i] = static_cast<parent>(i);
    });

    auto de = internal::dense_to_easy(n, edges, params, std::move(P));
    auto parents =
        internal::easy_case_from(n, de.root_edges, std::move(de.parents));

    gbbs::parallel_for(0, n, [&](size_t i) {
      internal::find_root(parents, static_cast<uintE>(i));
    });

    return parents;
  }

  sequence<parent> P_sparse(n);
  gbbs::parallel_for(0, n, [&](size_t i) {
    P_sparse[i] = static_cast<parent>(i);
  });

  sequence<uintE> V;
  sequence<internal::Edge> E;

  std::cerr << "[gazit] edges before sparse_to_dense: " << G.m << std::endl;

  std::cerr << "[gazit] vertices before sparse_to_dense: " << G.n
            << std::endl;

  auto dense_inputs = internal::sparse_to_dense(G, P_sparse);
  V = std::move(dense_inputs.first);
  E = std::move(dense_inputs.second);


  sequence<int> v_map(n, -1);
  gbbs::parallel_for(0, V.size(), [&](size_t i) {
    v_map[V[i]] = static_cast<int>(i);
  });
  size_t n2 = V.size();

  sequence<bool> keep(E.size());
  gbbs::parallel_for(0, E.size(), [&](size_t i){
    auto [u, v] = E[i];
    keep[i] = (u != v) && (v_map[u] >= 0) && (v_map[v] >= 0);
  });
  auto E2 = parlay::map(parlay::pack(E, keep), [&](internal::Edge e){
    return internal::Edge{static_cast<uintE>(v_map[e.first]),
                          static_cast<uintE>(v_map[e.second])};
  });

  std::cerr << "[gazit] edges before dense_to_easy: "
            << E2.size() << std::endl;
  std::cerr << "[gazit] vertices before dense_to_easy: "
            << n2 << std::endl;

  sequence<parent> P2(n2);
  gbbs::parallel_for(0, n2, [&](size_t i) { P2[i] = static_cast<parent>(i);});

  auto de = internal::dense_to_easy(n2, E2, params, std::move(P2));

  std::cerr << "[gazit] edges before easy_case: "
            << de.root_edges.size() << std::endl;
            
  auto P_small = internal::easy_case_from(n2, de.root_edges, std::move(de.parents));

  gbbs::parallel_for(0, n2, [&](size_t i) {
    internal::find_root(P_small, static_cast<uintE>(i));
  });

  sequence<parent> P_out(n);
  gbbs::parallel_for(0, n, [&](size_t i) {
    parent sparse_root = P_sparse[i];
    parent compact_root = P_small[static_cast<size_t>(v_map[sparse_root])];
    P_out[i] = V[static_cast<size_t>(compact_root)];
  });

  return P_out;
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

  std::cerr << "[comparison] Canonicalizing WorkEfficient labels" << std::endl;
  timer work_canon_timer;
  work_canon_timer.start();
  auto work_labels = internal::canonicalize_labels(work_components);
  double work_canon_time = work_canon_timer.stop();
  std::cerr << "[comparison] WorkEfficient canonicalization completed in "
            << work_canon_time << "s" << std::endl;

  std::cerr << "[comparison] Canonicalizing Gazit labels" << std::endl;
  timer gazit_canon_timer;
  gazit_canon_timer.start();
  auto gazit_labels = internal::canonicalize_labels(gazit_components);
  double gazit_canon_time = gazit_canon_timer.stop();
  std::cerr << "[comparison] Gazit canonicalization completed in "
            << gazit_canon_time << "s" << std::endl;
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
