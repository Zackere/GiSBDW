// Copyright 2020 GISBDW. All rights reserved.
#pragma once

#include <memory>
#include <utility>

#include "../branch_and_bound/branch_and_bound.hpp"
#include "../elimination_tree/elimination_tree.hpp"

namespace td {
class HighestDegreeHeuristic : public BranchAndBound::Heuristic {
 public:
  explicit HighestDegreeHeuristic(
      std::unique_ptr<BranchAndBound::Heuristic> heuristic)
      : BranchAndBound::Heuristic(std::move(heuristic)) {}
  ~HighestDegreeHeuristic() override = default;

  EliminationTree::Result Get(BranchAndBound::Graph const& g) override {
    EliminationTree tree(g);

    while (tree.ComponentsBegin() != tree.ComponentsEnd()) {
      auto const& g = tree.ComponentsBegin()->AdjacencyList();
      auto g_iter = g.begin();
      auto v = g_iter->first;
      auto best_degree = g_iter->second.size();

      while (++g_iter != g.end()) {
        if (g_iter->second.size() > best_degree) {
          v = g_iter->first;
          best_degree = g_iter->second.size();
        }
      }
      tree.Eliminate(v);
    }

    auto result = tree.Decompose();

    if (auto* h = Heuristic::Get()) {
      auto prev_result = h->Get(g);
      if (prev_result.treedepth < result.treedepth)
        return prev_result;
    }
    return result;
  }
};
}  // namespace td
