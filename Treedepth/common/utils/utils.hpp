// Copyright 2020 GISBDW. All rights reserved.
#pragma once

#include <algorithm>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <iostream>
#include <set>

#include "src/branch_and_bound/branch_and_bound.hpp"
#include "src/elimination_tree/elimination_tree.hpp"

namespace {  // NOLINT
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
template <typename EnterCallback, typename ExitCallback>
void DFS(int vertex,
         td::EliminationTree::BoostGraph const& g,
         std::set<int>* visited,
         EnterCallback enter,
         ExitCallback exit) {
  if (visited->find(vertex) != std::end(*visited))
    return;
  visited->insert(vertex);
  enter(vertex);
  for (auto [ai, ai_end] = boost::adjacent_vertices(vertex, g);  // NOLINT
       ai != ai_end; ++ai) {
    DFS(*ai, g, visited, enter, exit);
  }
  exit(vertex);
}

bool CheckIfTdDecompIsValid(td::BranchAndBound::Graph const& graph,
                            td::EliminationTree::Result const& result) {
  for (int v = 0; v < boost::num_vertices(graph); ++v) {
    for (auto [ai, ai_end] = boost::adjacent_vertices(v, graph);  // NOLINT
         ai != ai_end; ++ai) {
      std::set<int> tmp = {};
      int x = 0;
      bool ok = false;
      DFS(result.root, result.td_decomp, &tmp,
          [&](auto current_vertex) {
            if (current_vertex == v || current_vertex == *ai) {
              if (++x == 2)
                ok = true;
            }
          },
          [&](auto current_vertex) {
            if (current_vertex == v || current_vertex == *ai) {
              --x;
            }
          });
      if (!ok)
        return false;
    }
  }
  std::set<int> tmp = {};
  unsigned depth = 0;
  unsigned max_depth = 0;
  DFS(result.root, result.td_decomp, &tmp,
      [&](auto) {
        ++depth;
        max_depth = std::max(max_depth, depth);
      },
      [&](auto) { --depth; });
  return result.treedepth == max_depth;
}
}  // namespace
