// Copyright 2020 GISBDW. All rights reserved.
#pragma once

#include <boost/graph/adjacency_list.hpp>

#include "../../src/branch_and_bound/branch_and_bound.hpp"
#include "../../src/elimination_tree/elimination_tree.hpp"

namespace {
template <typename... Args>
bool CompareBoostGraphs(boost::adjacency_list<Args...> const& g1,
                        boost::adjacency_list<Args...> const& g2) {
  if (boost::num_vertices(g1) != boost::num_vertices(g2))
    return false;
  for (int i = 0; i < boost::num_vertices(g1); ++i)
    for (int j = 0; j < i; ++j)
      if (boost::edge(i, j, g1).second != boost::edge(i, j, g2).second)
        return false;
  return true;
}

bool CheckIfTdDecompIsValid(td::BranchAndBound::Graph const& graph,
                            td::EliminationTree::Result const& actual,
                            td::EliminationTree::Result const& expected) {
  // TODO(replinw): Implement it correctly
  return CompareBoostGraphs(actual.td_decomp, expected.td_decomp) &&
         actual.root == expected.root && actual.treedepth == expected.treedepth;
}
}  // namespace
