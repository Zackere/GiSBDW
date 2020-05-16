// Copyright 2020 GISBDW. All rights reserved.

#include <gtest/gtest.h>

#include "../../src/branch_and_bound/branch_and_bound.hpp"
#include "../../src/heuristics/highest_degree_heuristic.hpp"
#include "../../src/lower_bound/basic_lower_bound.hpp"
#include "../../src/lower_bound/edge_lower_bound.hpp"
#include "../../test/utils/graph_gen.hpp"

namespace {
class BCF : public ::testing::TestWithParam<Graph> {};
}  // namespace

TEST_P(BCF, BranchAndBoundBasicLoweBoundHighestDegreeHeuristic) {
  td::BranchAndBound bnb;
  bnb(GetParam(), std::make_unique<td::BasicLowerBound>(),
      std::make_unique<td::HighestDegreeHeuristic>(nullptr));
}

TEST_P(BCF, BranchAndBoundEdgeLoweBoundHighestDegreeHeuristic) {
  td::BranchAndBound bnb;
  bnb(GetParam(), std::make_unique<td::EdgeLowerBound>(),
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
