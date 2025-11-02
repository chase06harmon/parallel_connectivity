#include "benchmarks/Connectivity/Gazit1991/Connectivity.h"
#include "benchmarks/Connectivity/WorkEfficientSDB14/Connectivity.h"

#include <cstdlib>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>

#include "gbbs/graph.h"
#include "gbbs/graph_io.h"
#include "gbbs/macros.h"
#include "gbbs/unit_tests/graph_test_utils.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using ::testing::AnyOf;
using ::testing::ElementsAre;

namespace gbbs {

namespace {

std::vector<uintE> Canonicalize(const sequence<uintE>& labels) {
  std::vector<uintE> canonical(labels.size());
  std::unordered_map<uintE, uintE> remap;
  uintE next_id = 0;
  for (size_t i = 0; i < labels.size(); ++i) {
    auto it = remap.find(labels[i]);
    if (it == remap.end()) {
      it = remap.emplace(labels[i], next_id++).first;
    }
    canonical[i] = it->second;
  }
  return canonical;
}

std::string ResolveRunfile(const std::string& rel_path) {
  const char* srcdir = std::getenv("TEST_SRCDIR");
  const char* workspace = std::getenv("TEST_WORKSPACE");
  if (srcdir == nullptr || workspace == nullptr) {
    ADD_FAILURE() << "Runfile resolution failed: TEST_SRCDIR=" << srcdir
                  << " TEST_WORKSPACE=" << workspace;
    return rel_path;
  }
  return std::string(srcdir) + "/" + workspace + "/" + rel_path;
}

}  // namespace

TEST(WorkEfficientCC, EdgelessGraph) {
  constexpr uintE kNumVertices{3};
  const std::unordered_set<UndirectedEdge> kEdges{};
  auto graph{graph_test::MakeUnweightedSymmetricGraph(kNumVertices, kEdges)};

  const sequence<uintE> ccResult{workefficient_cc::CC(graph)};
  EXPECT_THAT(ccResult, ElementsAre(0, 1, 2));
}

TEST(WorkEfficientCC, BasicUsage) {
  // Graph diagram:
  //     0 - 1    2 - 3 - 4
  //                    \ |
  //                      5 -- 6
  constexpr uintE kNumVertices{7};
  const std::unordered_set<UndirectedEdge> kEdges{
      {0, 1}, {2, 3}, {3, 4}, {3, 5}, {4, 5}, {5, 6},
  };
  auto graph{graph_test::MakeUnweightedSymmetricGraph(kNumVertices, kEdges)};

  const sequence<uintE> ccResult{workefficient_cc::CC(graph)};
  uintE class_one = ccResult[0];
  uintE class_two = ccResult[2];
  EXPECT_THAT(ccResult, ElementsAre(class_one, class_one, class_two, class_two,
                                    class_two, class_two, class_two));
  EXPECT_NE(class_one, class_two);

  const auto gazit_result = gazit_cc::CC(graph);
  EXPECT_EQ(Canonicalize(ccResult), Canonicalize(gazit_result));
}

TEST(GazitCC, RandomGraphsMatchWorkEfficient) {
  std::mt19937_64 rng(189);
  std::uniform_int_distribution<int> vertex_count_dist(1, 8);
  std::uniform_real_distribution<double> edge_prob_dist(0.0, 1.0);

  for (int trial = 0; trial < 40; ++trial) {
    uintE n = static_cast<uintE>(vertex_count_dist(rng) + 2);
    double p = edge_prob_dist(rng);
    std::unordered_set<UndirectedEdge> edges;
    std::uniform_real_distribution<double> coin(0.0, 1.0);
    for (uintE u = 0; u < n; ++u) {
      for (uintE v = u + 1; v < n; ++v) {
        if (coin(rng) < p) {
          edges.emplace(u, v);
        }
      }
    }

    auto graph{graph_test::MakeUnweightedSymmetricGraph(n, edges)};
    const auto work_res = workefficient_cc::CC(graph);
    const auto gazit_res = gazit_cc::CC(graph);

    EXPECT_EQ(Canonicalize(work_res), Canonicalize(gazit_res))
        << "Trial " << trial << " with n = " << n << " and p = " << p;
  }
}

TEST(GazitCC, ProvidedGraphsMatchWorkEfficient) {
  const std::vector<std::string> paths = {
      "inputs/rMatGraph_J_5_100",
      "inputs/rMatGraph_WJ_5_100",
      "inputs/star.txt",
      "inputs/triangles.txt",
  };

  for (const auto& runfile : paths) {
    const std::string path = ResolveRunfile(runfile);
    ASSERT_FALSE(path.empty());
    auto graph = gbbs::gbbs_io::read_unweighted_symmetric_graph(
        path.c_str(), /*mmap=*/false, /*binary=*/false);
    auto work_res = workefficient_cc::CC(graph);
    auto gazit_res = gazit_cc::CC(graph);

    EXPECT_EQ(Canonicalize(work_res), Canonicalize(gazit_res)) << path;
  }
}

}  // namespace gbbs
