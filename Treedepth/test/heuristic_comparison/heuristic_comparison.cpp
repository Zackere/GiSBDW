// Copyright 2020 GISBDW. All rights reserved.

#include <gtest/gtest.h>

#include "../../src/branch_and_bound/branch_and_bound.hpp"
#include "../../src/heuristics/highest_degree_heuristic.hpp"
#include "../../src/heuristics/variance_heuristic.hpp"
#include "../../src/lower_bound/dynamic_gpu_lower_bound.hpp"
#include "../../src/lower_bound/edge_lower_bound.hpp"
#include "../utils/graph_gen.hpp"

namespace {
class BCF : public ::testing::TestWithParam<Graph> {};
}  // namespace

TEST_P(BCF, HeuristicComparison) {
  auto& g = GetParam();
  td::VarianceHeuristic v_heuristic(nullptr, 0.2, 0.2, 0.8);
  td::HighestDegreeHeuristic hd_heuristic(nullptr);

  std::cout << "Variance: " << v_heuristic.Get(g).treedepth << std::endl
            << "Highest degree: " << hd_heuristic.Get(g).treedepth << std::endl;

  // td::BranchAndBound bnb;
  // bnb(g,
  //     std::make_unique<td::DynamicGPULowerBound>(
  //         g, std::make_unique<td::EdgeLowerBound>()),
  //     std::make_unique<td::HighestDegreeHeuristic>(
  //         std::make_unique<td::VarianceHeuristic>(nullptr, 0.2, 0.2, 0.8)));
}

// INSTANTIATE_TEST_SUITE_P(Paths,
//                          BCF,
//                          ::testing::Values(Path(28),
//                                            Path(30),
//                                            Path(32),
//                                            Path(34),
//                                            Path(16),
//                                            Path(18),
//                                            Path(20),
//                                            Path(22),
//                                            Path(24),
//                                            Path(26)));
// INSTANTIATE_TEST_SUITE_P(ChordalCycles,
//                          BCF,
//                          ::testing::Values(ChordalCycle(26),
//                                            ChordalCycle(27),
//                                            ChordalCycle(28),
//                                            ChordalCycle(29),
//                                            ChordalCycle(30),
//                                            ChordalCycle(41)));
//
// INSTANTIATE_TEST_SUITE_P(Cycles,
//                          BCF,
//                          ::testing::Values(Cycle(20),
//                                            Cycle(30),
//                                            Cycle(34),
//                                            Cycle(35),
//                                            Cycle(36),
//                                            Cycle(37),
//                                            Cycle(38),
//                                            Cycle(39),
//                                            Cycle(40),
//                                            Cycle(44)));
//
// INSTANTIATE_TEST_SUITE_P(SparseGraphs,
//                          BCF,
//                          ::testing::Values(RandomSparseConnectedGraph(22),
//                                            RandomSparseConnectedGraph(33),
//                                            RandomSparseConnectedGraph(34),
//                                            RandomSparseConnectedGraph(35),
//                                            RandomSparseConnectedGraph(36),
//                                            RandomSparseConnectedGraph(37),
//                                            RandomSparseConnectedGraph(38),
//                                            RandomSparseConnectedGraph(39),
//                                            RandomSparseConnectedGraph(40)));
//
INSTANTIATE_TEST_SUITE_P(HalinGraphs,
                         BCF,
                         ::testing::Values(Halin(27),
                                           Halin(30),
                                           Halin(32),
                                           Halin(34),
                                           Halin(36),
                                           Halin(38),
                                           Halin(40)));
