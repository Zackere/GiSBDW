// Copyright 2020 GISBDW. All rights reserved.
#pragma once

#include <memory>
#include <utility>

#include "src/branch_and_bound/branch_and_bound.hpp"
#include "src/elimination_tree/elimination_tree.hpp"

namespace td {
class HighestDegreeHeuristic : public BranchAndBound::Heuristic {
 public:
  explicit HighestDegreeHeuristic(
      std::unique_ptr<BranchAndBound::Heuristic> heuristic)
      : heuristic_(std::move(heuristic)) {}
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

    if (heuristic_) {
      auto prev_result = heuristic_->Get(g);
      if (prev_result.treedepth < result.treedepth)
        return prev_result;
    }
    return result;
  }

 private:
  std::unique_ptr<Heuristic> heuristic_;
};
}  // namespace td
