#include "benchmarks/Connectivity/Gazit1991/Connectivity.h"

namespace gbbs {
namespace gazit_cc {

std::mt19937& rng() {
    static thread_local std::mt19937 gen(std::random_device{}());
    return gen;
}

int random_int(int lo, int hi) {
    std::uniform_int_distribution<int> dist(lo, hi);
    return dist(rng());
}

void run_test_slow(size_t N) {
  sequence<uintE> V_roots(N);
  sequence<uintE> P(N);
  sequence<int> next(N, -1);
  sequence<int> mate_j(N, -1);
  int k = 0; 
  int j = 0; 

  gbbs::parallel_for(0, N, [&](size_t i){V_roots[i] = i; P[i] = i;});

  for (int i = 0; i < N; i++) {
    next[i] = random_int(0, N-1);
  }

  using clock = std::chrono::high_resolution_clock;

  auto t0 = clock::now();
  // NOTE: adjust args if your deterministic_mate signature is different
  internal::deterministic_mate(V_roots, P, next, mate_j, j);
  auto t1 = clock::now();

  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  double ns_per_vertex = (ms * 1e6) / static_cast<double>(N);

  std::cout << "size: " << N << "\t" << "time: " << ms << " ms" << std::endl;
  // std::cout << "ns per vertex: " << ns_per_vertex << " ns" << std::endl;
}

void run_test_fast(size_t N) {
  sequence<uintE> V_roots(N);
  sequence<uintE> P(N);
  sequence<int> next(N, -1);
  sequence<int> mate_j(N, -1);
  int k = 0; 
  int j = 0; 

  gbbs::parallel_for(0, N, [&](size_t i){V_roots[i] = i; P[i] = i;});

  for (int i = 0; i < N; i++) {
    next[i] = random_int(0, N-1);
  }

  using clock = std::chrono::high_resolution_clock;

  auto t0 = clock::now();
  // NOTE: adjust args if your deterministic_mate signature is different
  internal::deterministic_mate2(V_roots, P, next, mate_j, j);
  auto t1 = clock::now();

  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  double ns_per_vertex = (ms * 1e6) / static_cast<double>(N);

  std::cout << "size: " << N << "\t" << "time: " << ms << " ms" << std::endl;
  // std::cout << "ns per vertex: " << ns_per_vertex << " ns" << std::endl;
}

void test_deterministic_mate() {
  
  std::vector<size_t> slow_sizes = {
    1'000, 5'000, 10'000, 25'000, 50'000,
    100'000, 250'000, 500'000,
    1'000'000, 2'500'000
  };
  
  std::vector<size_t> fast_sizes = {
    1'000, 5'000, 10'000, 25'000, 50'000,
    100'000, 250'000, 500'000,
    1'000'000, 2'500'000, 5'000'000,
    7'500'000, 10'000'000, 15'000'000,
    20'000'000, 30'000'000, 50'000'000
  };

  std::cout << "TESTING SLOW" << std::endl;

  for (auto &N : slow_sizes) {
    run_test_slow(N);
  }

  std::cout << "TESTING FAST" << std::endl;

  for (auto &N : fast_sizes) {
    run_test_fast(N);
  }


  // std::unordered_set<int> roots;

  // for (int i = 0; i < N; i++) {
  //   roots.insert(P[i]);
  // }

  // std::cout << "num roots: " << roots.size() << std::endl;


  // for (auto & r : roots) {
  //   if (P[r] != r) {
  //     std::cout << "FRAUD" << std::endl;
  //   }
  // }

  // For correctness, needs to be all stars. How do we check? 
  // 

  // for (auto & v : path) {
  //   std::cout << "node: " << v << "\t" << "parent: " << std::endl;
  // }

  // gbbs::parallel_for(0, N, [&](size_t i){V_roots[i] = i; P[i] = i; next[i] = (i+1) % N;});


  // size_t N = 9;

  // sequence<uintE> V_roots(N);
  // sequence<uintE> P(N);
  // sequence<int> mate_j(N, -1);
  // int k = 0; 
  // int j = 0; 
  // int num_threads = N / log2(N);

  // gbbs::parallel_for(0, N, [&](size_t i){V_roots[i] = i; P[i] = i;});

  // sequence<int> next = {1, 2, 1, 4, 2, 3, 7, 8, 4};

  // std::cout << "Num threads: " << num_threads << std::endl;
  // std::cout << "Per thread: " << (N + num_threads - 1) / num_threads  << std::endl;

  // for (int i = 0; i < N; i++) {
  //   // if (P[i] == P[i+1]) {
  //     std::cout << "i: " << i << " p: " << P[i] << std::endl;
  //   // }
  // }

}

template<class Graph>
void test_sample_edges(Graph & G) {
  auto edges = parlay::map(G.edges(), [](const auto& entry) {
    uintE u, v; gbbs::empty _;
    std::tie(u, v, _) = entry;
    return internal::Edge{u, v};
  });

  auto new_edges = internal::sample_edges(edges, 125);
  std::cout << "Old Edge Size: " << edges.size() << std::endl;
  std::cout << "New Edge Size: " << new_edges.size() << std::endl;

}

}
}
