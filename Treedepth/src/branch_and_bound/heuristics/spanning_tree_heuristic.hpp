// Copyright 2020 GISBDW. All rights reserved.
#pragma once

#include "common/utils/graph_gen.hpp"
#include "src/branch_and_bound/branch_and_bound.hpp"
#include "src/elimination_tree/elimination_tree.hpp"
#include "src/union_find/array_union_find.hpp"

namespace td {
class SpanningTreeHeuristic : public BranchAndBound::Heuristic {
 public:
  explicit SpanningTreeHeuristic(
      std::unique_ptr<BranchAndBound::Heuristic> heuristic)
      : heuristic_(std::move(heuristic)) {}
  ~SpanningTreeHeuristic() override = default;
  EliminationTree::Result Get(BranchAndBound::Graph const& g) override {
    // we assume that g is a tree
    auto spanning_tree = SpanningTree(g);
    EliminationTree spanning_tree_elimination(spanning_tree);
    EliminationTree result(g);
    while (spanning_tree_elimination.ComponentsBegin() !=
           spanning_tree_elimination.ComponentsEnd()) {
      if (spanning_tree_elimination.ComponentsBegin()->AdjacencyList().size() <
          3) {
        result.Eliminate(
            std::begin(
                spanning_tree_elimination.ComponentsBegin()->AdjacencyList())
                ->first);
        spanning_tree_elimination.Eliminate(
            std::begin(
                spanning_tree_elimination.ComponentsBegin()->AdjacencyList())
                ->first);
        continue;
      }
      // take component and find its center
      std::set<int> taken_total;

      for (auto& p :
           spanning_tree_elimination.ComponentsBegin()->AdjacencyList())
        if (p.second.size() == 1)
          taken_total.insert(p.first);

      while (
          taken_total.size() + 2 <
          spanning_tree_elimination.ComponentsBegin()->AdjacencyList().size()) {
        std::set<int> taken_now;
        for (auto& p :
             spanning_tree_elimination.ComponentsBegin()->AdjacencyList()) {
          if (taken_total.find(p.first) != std::end(taken_total))
            continue;
          if (std::count_if(std::begin(p.second), std::end(p.second),
                            [&taken_total](auto v) {
                              return taken_total.find(v) ==
                                     std::end(taken_total);
                            }) == 1)
            taken_now.insert(p.first);
        }
        for (auto v : taken_now)
          taken_total.insert(v);
      }

      for (auto& p :
           spanning_tree_elimination.ComponentsBegin()->AdjacencyList()) {
        if (taken_total.find(p.first) == std::end(taken_total)) {
          result.Eliminate(p.first);
          spanning_tree_elimination.Eliminate(p.first);
          break;
        }
      }
    }
    auto decomposition = result.Decompose();

    if (heuristic_) {
      auto prev_result = heuristic_->Get(g);
      if (prev_result.treedepth < decomposition.treedepth)
        return prev_result;
    }
    return decomposition;
  }

 private:
  std::unique_ptr<BranchAndBound::Heuristic> heuristic_;
};
}  // namespace td
