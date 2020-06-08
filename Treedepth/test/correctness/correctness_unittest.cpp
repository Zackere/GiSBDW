// Copyright 2020 GISBDW. All rights reserved.
#include <gtest/gtest.h>

#include <boost/graph/graphviz.hpp>
#include <fstream>

#include "common/utils/graph_gen.hpp"
#include "common/utils/utils.hpp"
#include "src/branch_and_bound/branch_and_bound.hpp"
#include "src/branch_and_bound/heuristics/highest_degree_heuristic.hpp"
#include "src/branch_and_bound/lower_bound/basic_lower_bound.hpp"
#include "src/branch_and_bound/lower_bound/edge_lower_bound.hpp"
#include "src/dynamic_cpu/dynamic_cpu.hpp"
#include "src/dynamic_cpu/dynamic_cpu_improv.hpp"
#include "src/dynamic_gpu/dynamic_gpu.hpp"

namespace {
class CTF : public ::testing::TestWithParam<Graph> {};
time_t seed = time(0);
// time_t seed = 1589591285;

// time_t seed = 1589671678;
// time_t seed = 1589671728;
// time_t seed = 1589671906;
// time_t seed = 1589671960;
// time_t seed = 1589672194;
// time_t seed = 1589729310;
}  // namespace

// TEST_P(CTF, CorrectnessTest) {
//   auto const& g = GetParam();
//   td::BranchAndBound bnb;
//   auto res_bnb = bnb(g, std::make_unique<td::EdgeLowerBound>(),
//                      std::make_unique<td::HighestDegreeHeuristic>(nullptr));
//   EXPECT_TRUE(CheckIfTdDecompIsValid(g, res_bnb));
//
//   td::DynamicCPUImprov dyncpu_improv;
//   dyncpu_improv(g);
//   td::DynamicCPUImprov::CodeType code = (1 << boost::num_vertices(g)) - 1;
//   auto res_dyncpu_imrprov = dyncpu_improv.GetTDDecomp(code, g);
//   EXPECT_TRUE(CheckIfTdDecompIsValid(g, res_dyncpu_imrprov));
//   EXPECT_EQ(res_bnb.treedepth, res_dyncpu_imrprov.treedepth);
//
//   td::DynamicCPU dyncpu;
//   dyncpu(g);
//   auto res_dyncpu = dyncpu.GetTDDecomp(0, g);
//   EXPECT_TRUE(CheckIfTdDecompIsValid(g, res_dyncpu));
//   EXPECT_EQ(res_bnb.treedepth, res_dyncpu.treedepth);
// }
TEST_P(CTF, idk) {
  td::DynamicGPU dyngpu;
  dyngpu(GetParam());
  int n = boost::num_vertices(GetParam());
  auto el = dyngpu.GetElimination<td::EliminationTree::VertexType>(n, n, 0);
  td::EliminationTree eltree(GetParam());
  for (auto v : el)
    eltree.Eliminate(v);
  auto tddecomp = eltree.Decompose();
  td::DynamicCPUImprov dyncpu_improv;
  dyncpu_improv(GetParam());
  td::DynamicCPUImprov::CodeType code =
      (1 << boost::num_vertices(GetParam())) - 1;
  auto res_dyncpu_imrprov = dyncpu_improv.GetTDDecomp(code, GetParam());
  EXPECT_EQ(tddecomp.treedepth, res_dyncpu_imrprov.treedepth);
}
INSTANTIATE_TEST_SUITE_P(Paths,
                         CTF,
                         ::testing::Values(Path(7),
                                           Path(8),
                                           Path(10),
                                           Path(12),
                                           Path(14),
                                           Path(16),
                                           Path(21)));

INSTANTIATE_TEST_SUITE_P(ChordalCycles,
                         CTF,
                         ::testing::Values(ChordalCycle(10),
                                           ChordalCycle(11),
                                           ChordalCycle(12),
                                           ChordalCycle(13),
                                           ChordalCycle(14),
                                           ChordalCycle(15)));

INSTANTIATE_TEST_SUITE_P(Cycles,
                         CTF,
                         ::testing::Values(Cycle(7),
                                           Cycle(8),
                                           Cycle(9),
                                           Cycle(10),
                                           Cycle(11),
                                           Cycle(12),
                                           Cycle(13),
                                           Cycle(14),
                                           Cycle(15),
                                           Cycle(16)));

INSTANTIATE_TEST_SUITE_P(SparseGraphs,
                         CTF,
                         ::testing::Values(RandomSparseConnectedGraph(15, seed),
                                           RandomSparseConnectedGraph(16, seed),
                                           RandomSparseConnectedGraph(17, seed),
                                           RandomSparseConnectedGraph(18, seed),
                                           RandomSparseConnectedGraph(10, seed),
                                           RandomSparseConnectedGraph(9, seed),
                                           RandomSparseConnectedGraph(8,
                                                                      seed)));

INSTANTIATE_TEST_SUITE_P(SparseGraphs2,
                         CTF,
                         ::testing::Values(RandomSparseConnectedGraph(14, seed),
                                           RandomSparseConnectedGraph(15,
                                                                      seed)));
