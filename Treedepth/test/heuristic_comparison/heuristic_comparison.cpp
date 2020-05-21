// Copyright 2020 GISBDW. All rights reserved.

#include <gtest/gtest.h>

#include <boost/graph/graphviz.hpp>
#include <fstream>

#include "common/utils/graph_gen.hpp"
#include "src/branch_and_bound/branch_and_bound.hpp"
#include "src/branch_and_bound/heuristics/bottom_up_heuristic.hpp"
#include "src/branch_and_bound/heuristics/highest_degree_heuristic.hpp"
#include "src/branch_and_bound/heuristics/spanning_tree_heuristic.hpp"
#include "src/branch_and_bound/heuristics/variance_heuristic.hpp"
#include "src/branch_and_bound/lower_bound/edge_lower_bound.hpp"

namespace {
class BCF : public ::testing::TestWithParam<Graph> {};
}  // namespace

TEST_P(BCF, HeuristicComparison) {
  auto& g = GetParam();
  td::VarianceHeuristic v_heuristic(nullptr, 0.2, 0.2, 0.8);
  td::HighestDegreeHeuristic hd_heuristic(nullptr);
  td::SpanningTreeHeuristic st_heuristic(nullptr);
  td::BottomUpHeuristicGPU bu_heuristic(nullptr);
  td::BranchAndBound bnb;
  auto res_bnb =
      bnb(g, std::make_unique<td::EdgeLowerBound>(),
          std::make_unique<td::HighestDegreeHeuristic>(
              std::make_unique<td::VarianceHeuristic>(
                  std::make_unique<td::BottomUpHeuristicGPU>(nullptr), 1.0, 0.2,
                  0.8)));

  std::cout << "Variance: " << v_heuristic.Get(g).treedepth << std::endl
            << "Highest degree: " << hd_heuristic.Get(g).treedepth << std::endl
            << "Spanning tree heuristic: " << st_heuristic.Get(g).treedepth
            << std::endl
            << "Akka dynamic heuristic: " << bu_heuristic.Get(g).treedepth
            << std::endl
            << "Actual treedepth: " << res_bnb.treedepth << std::endl;
}
INSTANTIATE_TEST_SUITE_P(Paths,
                         BCF,
                         ::testing::Values(Path(16),
                                           Path(18),
                                           Path(20),
                                           Path(22),
                                           Path(24),
                                           Path(26)));
INSTANTIATE_TEST_SUITE_P(ChordalCycles,
                         BCF,
                         ::testing::Values(ChordalCycle(26),
                                           ChordalCycle(21),
                                           ChordalCycle(24),
                                           ChordalCycle(25),
                                           ChordalCycle(20),
                                           ChordalCycle(21)));

INSTANTIATE_TEST_SUITE_P(Cycles,
                         BCF,
                         ::testing::Values(Cycle(20),
                                           Cycle(25),
                                           Cycle(27),
                                           Cycle(14),
                                           Cycle(26),
                                           Cycle(27),
                                           Cycle(28),
                                           Cycle(29),
                                           Cycle(30),
                                           Cycle(10)));

INSTANTIATE_TEST_SUITE_P(SparseGraphs,
                         BCF,
                         ::testing::Values(RandomSparseConnectedGraph(22),
                                           RandomSparseConnectedGraph(25),
                                           RandomSparseConnectedGraph(24),
                                           RandomSparseConnectedGraph(35),
                                           RandomSparseConnectedGraph(36),
                                           RandomSparseConnectedGraph(37),
                                           RandomSparseConnectedGraph(38),
                                           RandomSparseConnectedGraph(29),
                                           RandomSparseConnectedGraph(30)));

INSTANTIATE_TEST_SUITE_P(SmallSparseGraphs,
                         BCF,
                         ::testing::Values(RandomSparseConnectedGraph(10),
                                           RandomSparseConnectedGraph(11),
                                           RandomSparseConnectedGraph(12),
                                           RandomSparseConnectedGraph(13),
                                           RandomSparseConnectedGraph(14),
                                           RandomSparseConnectedGraph(15),
                                           RandomSparseConnectedGraph(16),
                                           RandomSparseConnectedGraph(17),
                                           RandomSparseConnectedGraph(18)));

INSTANTIATE_TEST_SUITE_P(HalinGraphs,
                         BCF,
                         ::testing::Values(Halin(7),
                                           Halin(10),
                                           Halin(12),
                                           Halin(14),
                                           Halin(15),
                                           Halin(23),
                                           Halin(27)));
