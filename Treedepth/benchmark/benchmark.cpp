// Copyright 2020 GISBDW. All rights reserved.

#include <cmath>
#include <fstream>
#include <iostream>
#include <random>

#include "../src/branch_and_bound/branch_and_bound.hpp"
#include "../src/heuristics/highest_degree_heuristic.hpp"
#include "../src/lower_bound/basic_lower_bound.hpp"
#include "../src/union_find/array_union_find.hpp"
#include "boost/graph/adjacency_list.hpp"
#include "boost/graph/erdos_renyi_generator.hpp"
#include "boost/graph/graphviz.hpp"
#include "gtest/gtest.h"

namespace {
using Graph =
    boost::adjacency_list<boost::mapS, boost::vecS, boost::undirectedS>;
using ERGen = boost::sorted_erdos_renyi_iterator<std::minstd_rand, Graph>;
class BCF : public ::testing::TestWithParam<Graph> {};
Graph RandomConnectedGraph(int n, float p, int seed = 0) {
  Graph ret;
  do {
    std::minstd_rand rng(++seed);
    ret = Graph(ERGen(rng, n, p), ERGen(), n);
  } while (boost::connected_components(
               ret, std::vector<int>(boost::num_vertices(ret)).data()) != 1);
  return ret;
}
Graph RandomSparseConnectedGraph(int n) {
  return RandomConnectedGraph(n, std::log(n) / n);
}
Graph Path(int n) {
  Graph g(n);
  for (std::size_t i = 0; i < n - 1; ++i)
    boost::add_edge(i, i + 1, g);
  return g;
}
Graph Cycle(int n) {
  Graph g = Path(n);
  boost::add_edge(0, n - 1, g);
  return g;
}
Graph Complete(int n) {
  Graph g(n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < i; ++j)
      boost::add_edge(i, j, g);
  return g;
}
Graph SpanningTree(Graph g) {
  Graph ret(boost::num_vertices(g));
  td::ArrayUnionFind<int> uf(boost::num_vertices(g));
  typename boost::graph_traits<
      std::remove_reference_t<decltype(g)>>::out_edge_iterator ei,
      ei_end;
  for (int i = 0; i < boost::num_vertices(g); ++i)
    for (boost::tie(ei, ei_end) = out_edges(i, g); ei != ei_end; ++ei) {
      auto idi = uf.Find(i);
      auto idx = uf.Find(boost::target(*ei, g));
      if (idi != idx) {
        uf.Union(idi, idx);
        boost::add_edge(i, boost::target(*ei, g), ret);
      }
    }
  return ret;
}
Graph Halin(int n) {
  Graph g = RandomSparseConnectedGraph(n);
  std::vector<boost::graph_traits<Graph>::edge_descriptor> p(n);
  g = SpanningTree(g);
  std::vector<boost::graph_traits<Graph>::vertex_descriptor> leaves;
  leaves.reserve(n);
  for (std::size_t i = 0; i < p.size(); ++i)
    if (boost::out_degree(i, g) == 1)
      leaves.emplace_back(i);
  for (std::size_t i = 0; i < leaves.size() - 1; ++i)
    boost::add_edge(leaves[i], leaves[i + 1], g);
  boost::add_edge(leaves[0], leaves[leaves.size() - 1], g);
  return g;
}
Graph ChordalCycle(int n) {
  Graph g = Cycle(n);
  boost::add_edge(0, n / 3, g);
  boost::add_edge(0, 2 * n / 3, g);
  return g;
}
}  // namespace

TEST_P(BCF, BranchAndBoundBasicLoweBoundHighestDegreeHeuristic) {
  td::BranchAndBound bnb;
  bnb(GetParam(), std::make_unique<td::BasicLowerBound>(),
      std::make_unique<td::HighestDegreeHeuristic>(nullptr));
}

INSTANTIATE_TEST_SUITE_P(Paths,
                         BCF,
                         ::testing::Values(Path(8),
                                           Path(10),
                                           Path(12),
                                           Path(14),
                                           Path(16),
                                           Path(18),
                                           Path(20),
                                           Path(22),
                                           Path(24),
                                           Path(26)));
INSTANTIATE_TEST_SUITE_P(Cycles,
                         BCF,
                         ::testing::Values(Cycle(8),
                                           Cycle(10),
                                           Cycle(12),
                                           Cycle(14),
                                           Cycle(16),
                                           Cycle(18),
                                           Cycle(20),
                                           Cycle(22),
                                           Cycle(24),
                                           Cycle(26)));
INSTANTIATE_TEST_SUITE_P(ChordalCycles,
                         BCF,
                         ::testing::Values(ChordalCycle(8),
                                           ChordalCycle(10),
                                           ChordalCycle(12),
                                           ChordalCycle(14),
                                           ChordalCycle(16),
                                           ChordalCycle(18),
                                           ChordalCycle(20),
                                           ChordalCycle(22),
                                           ChordalCycle(24),
                                           ChordalCycle(26)));
INSTANTIATE_TEST_SUITE_P(
    Trees,
    BCF,
    ::testing::Values(SpanningTree(RandomSparseConnectedGraph(8)),
                      SpanningTree(RandomSparseConnectedGraph(10)),
                      SpanningTree(RandomSparseConnectedGraph(12)),
                      SpanningTree(RandomSparseConnectedGraph(14)),
                      SpanningTree(RandomSparseConnectedGraph(16)),
                      SpanningTree(RandomSparseConnectedGraph(18)),
                      SpanningTree(RandomSparseConnectedGraph(20)),
                      SpanningTree(RandomSparseConnectedGraph(22)),
                      SpanningTree(RandomSparseConnectedGraph(24)),
                      SpanningTree(RandomSparseConnectedGraph(26)),
                      SpanningTree(RandomSparseConnectedGraph(28)),
                      SpanningTree(RandomSparseConnectedGraph(30))));
INSTANTIATE_TEST_SUITE_P(HalinGraphs,
                         BCF,
                         ::testing::Values(Halin(8),
                                           Halin(10),
                                           Halin(12),
                                           Halin(14),
                                           Halin(16),
                                           Halin(18),
                                           Halin(20)));
INSTANTIATE_TEST_SUITE_P(SparseGraphs,
                         BCF,
                         ::testing::Values(RandomSparseConnectedGraph(8),
                                           RandomSparseConnectedGraph(10),
                                           RandomSparseConnectedGraph(12),
                                           RandomSparseConnectedGraph(14),
                                           RandomSparseConnectedGraph(16),
                                           RandomSparseConnectedGraph(18),
                                           RandomSparseConnectedGraph(20)));
INSTANTIATE_TEST_SUITE_P(Complete,
                         BCF,
                         ::testing::Values(Complete(2),
                                           Complete(3),
                                           Complete(4),
                                           Complete(5),
                                           Complete(6),
                                           Complete(7),
                                           Complete(8),
                                           Complete(9),
                                           Complete(10)));
