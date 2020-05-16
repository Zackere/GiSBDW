#include <gtest/gtest.h>

#include <boost/graph/graphviz.hpp>
#include <fstream>

#include "../../src/branch_and_bound/branch_and_bound.hpp"
#include "../../src/dynamic_gpu/dynamic_gpu.hpp"
#include "../../src/heuristics/highest_degree_heuristic.hpp"
#include "../../src/lower_bound/edge_lower_bound.hpp"
#include "../graph_gen.hpp"

namespace {
class CTF : public ::testing::TestWithParam<Graph> {};
time_t seed = 1589591285;
}  // namespace

TEST_P(CTF, CorrectnessTest) {
  auto const& g = GetParam();
  td::BranchAndBound bnb;
  auto res_bnb = bnb(g, std::make_unique<td::EdgeLowerBound>(),
                     std::make_unique<td::HighestDegreeHeuristic>(nullptr));
  td::DynamicGPU dgpu;
  dgpu(g);
  EXPECT_EQ(res_bnb.treedepth, dgpu.GetTreedepth(boost::num_vertices(g),
                                                 boost::num_vertices(g), 0));
  td::EliminationTree et(GetParam());
  for (auto v : dgpu.GetElimination<td::EliminationTree::VertexType>(
           boost::num_vertices(g), boost::num_vertices(g), 0))
    et.Eliminate(v);
  auto res_dyngpu = et.Decompose();
  EXPECT_EQ(res_bnb.treedepth, res_dyngpu.treedepth);
}

// INSTANTIATE_TEST_SUITE_P(Paths,
//                         CTF,
//                         ::testing::Values(Path(5),
//                                           Path(6),
//                                           Path(7),
//                                           Path(8),
//                                           Path(9),
//                                           Path(10),
//                                           Path(11),
//                                           Path(12),
//                                           Path(13),
//                                           Path(14),
//                                           Path(15)));
//
// INSTANTIATE_TEST_SUITE_P(ChordalCycles,
//                         CTF,
//                         ::testing::Values(ChordalCycle(10),
//                                           ChordalCycle(11),
//                                           ChordalCycle(12),
//                                           ChordalCycle(13),
//                                           ChordalCycle(14),
//                                           ChordalCycle(15)));
//
// INSTANTIATE_TEST_SUITE_P(Cycles,
//                         CTF,
//                         ::testing::Values(Cycle(7),
//                                           Cycle(8),
//                                           Cycle(9),
//                                           Cycle(10),
//                                           Cycle(11),
//                                           Cycle(12),
//                                           Cycle(13),
//                                           Cycle(14),
//                                           Cycle(15),
//                                           Cycle(16)));

INSTANTIATE_TEST_SUITE_P(SparseGraphs,
                         CTF,
                         ::testing::Values(RandomSparseConnectedGraph(15,
                                                                      seed)));
