// Copyright 2020 GISBDW. All rights reserved.

#include <gtest/gtest.h>

#include "common/utils/graph_gen.hpp"
#include "src/bnb_gpu/bnb_gpu.hpp"
#include "src/branch_and_bound/branch_and_bound.hpp"
#include "src/branch_and_bound/heuristics/highest_degree_heuristic.hpp"
#include "src/branch_and_bound/heuristics/variance_heuristic.hpp"
#include "src/branch_and_bound/lower_bound/basic_lower_bound.hpp"
#include "src/branch_and_bound/lower_bound/edge_lower_bound.hpp"
#include "src/dynamic_cpu/dynamic_cpu.hpp"
#include "src/dynamic_cpu/dynamic_cpu_improv.hpp"
#include "src/dynamic_gpu/dynamic_gpu.hpp"

namespace {
class BCF : public ::testing::TestWithParam<Graph> {};
}  // namespace

// TEST_P(BCF, DynamicCPU) {
//  td::DynamicCPU dyncpu;
//  dyncpu(GetParam());
//  dyncpu.GetTDDecomp(0, GetParam());
//}

// TEST_P(BCF, DynamicCPUImprov) {
//  td::DynamicCPUImprov dyncpu;
//  dyncpu(GetParam());
//  td::DynamicCPUImprov::CodeType code = 1;
//  code <<= boost::num_vertices(GetParam());
//  dyncpu.GetTDDecomp(--code, GetParam());
//}

// TEST_P(BCF, BNBBasicLoweBoundHighestDegreeHeuristic) {
//  td::BranchAndBound bnb;
//  bnb(GetParam(), std::make_unique<td::BasicLowerBound>(),
//      std::make_unique<td::HighestDegreeHeuristic>(nullptr));
//}
//
// TEST_P(BCF, BNBEdgeLoweBoundHighestDegreeHeuristic) {
//  td::BranchAndBound bnb;
//  bnb(GetParam(), std::make_unique<td::EdgeLowerBound>(),
//      std::make_unique<td::HighestDegreeHeuristic>(
//          std::make_unique<td::VarianceHeuristic>(nullptr, 1.0, 0.2, 0.8)));
//}

TEST_P(BCF, DynGPU) {
  // td::DynamicGPU dyngpu;
  // dyngpu(GetParam());
  td::BnBGPU bnb;
  bnb(GetParam(), boost::num_vertices(GetParam()) + 1);
}

// INSTANTIATE_TEST_SUITE_P(Complete,
//                         BCF,
//                         ::testing::Values(Complete(5),
//                                           Complete(6),
//                                           Complete(7),
//                                           Complete(8),
//                                           Complete(9),
//                                           Complete(10),
//                                           Complete(11),
//                                           Complete(12),
//                                           Complete(13)));
//
// INSTANTIATE_TEST_SUITE_P(
//    Paths,
//    BCF,
//    ::testing::Values(Path(8), Path(10), Path(12), Path(14), Path(16)));
// INSTANTIATE_TEST_SUITE_P(Cycles,
//                         BCF,
//                         ::testing::Values(Cycle(8),
//                                           Cycle(10),
//                                           Cycle(12),
//                                           Cycle(14),
//                                           Cycle(16),
//                                           Cycle(18)));
// INSTANTIATE_TEST_SUITE_P(ChordalCycles,
//                         BCF,
//                         ::testing::Values(ChordalCycle(8),
//                                           ChordalCycle(10),
//                                           ChordalCycle(12),
//                                           ChordalCycle(14),
//                                           ChordalCycle(16),
//                                           ChordalCycle(18)));
// INSTANTIATE_TEST_SUITE_P(
//    Trees,
//    BCF,
//    ::testing::Values(SpanningTree(RandomSparseConnectedGraph(8)),
//                      SpanningTree(RandomSparseConnectedGraph(10)),
//                      SpanningTree(RandomSparseConnectedGraph(12)),
//                      SpanningTree(RandomSparseConnectedGraph(14)),
//                      SpanningTree(RandomSparseConnectedGraph(16))));

// INSTANTIATE_TEST_SUITE_P(HalinGraphs,
//                         BCF,
//                         ::testing::Values(Halin(8),
//                                           Halin(10),
//                                           Halin(12),
//                                           Halin(14),
//                                           Halin(16),
//                                           Halin(18),
//                                           Halin(20)));
// INSTANTIATE_TEST_SUITE_P(SparseGraphs,
//                         BCF,
//                         ::testing::Values(RandomSparseConnectedGraph(8),
//                                           RandomSparseConnectedGraph(10),
//                                           RandomSparseConnectedGraph(12),
//                                           RandomSparseConnectedGraph(14),
//                                           RandomSparseConnectedGraph(16)));

INSTANTIATE_TEST_SUITE_P(Test, BCF, ::testing::Values(Path(16)));
