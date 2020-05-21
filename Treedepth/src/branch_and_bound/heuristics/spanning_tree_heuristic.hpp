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
    EliminationTree et(spanning_tree);
    while (et.ComponentsBegin() != et.ComponentsEnd()) {
      if (et.ComponentsBegin()->AdjacencyList().size() < 3) {
        et.Eliminate(std::begin(et.ComponentsBegin()->AdjacencyList())->first);
        continue;
      }
      // take component and find its center
      std::set<int> taken_total;

      for (auto& p : et.ComponentsBegin()->AdjacencyList())
        if (p.second.size() == 1)
          taken_total.insert(p.first);

      while (taken_total.size() + 2 <
             et.ComponentsBegin()->AdjacencyList().size()) {
        std::set<int> taken_now;
        for (auto& p : et.ComponentsBegin()->AdjacencyList()) {
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

      for (auto& p : et.ComponentsBegin()->AdjacencyList()) {
        if (taken_total.find(p.first) == std::end(taken_total)) {
          et.Eliminate(p.first);
          break;
        }
      }
    }
    auto result = et.Decompose();

    if (heuristic_) {
      auto prev_result = heuristic_->Get(g);
      if (prev_result.treedepth < result.treedepth)
        return prev_result;
    }
    return result;
  }

 private:
  std::unique_ptr<BranchAndBound::Heuristic> heuristic_;
};
}  // namespace td
