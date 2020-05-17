// Copyright 2020 GISBDW. All rights reserved.
#pragma once

#include <cmath>
#include <memory>
#include <numeric>
#include <utility>

#include "../branch_and_bound/branch_and_bound.hpp"
#include "../elimination_tree/elimination_tree.hpp"

namespace td {
class VarianceHeuristic : public BranchAndBound::Heuristic {
 public:
  explicit VarianceHeuristic(
      std::unique_ptr<BranchAndBound::Heuristic> heuristic,
      float A,
      float D,
      float E)
      : heuristic_(std::move(heuristic)), A_(A), D_(D), E_(E) {}
  ~VarianceHeuristic() override = default;

  EliminationTree::Result Get(BranchAndBound::Graph const& graph) override {
    EliminationTree tree(graph);

    // as long as tree is not ready
    while (tree.ComponentsBegin() != tree.ComponentsEnd()) {
      double best_vertex_coef = -1;
      double current_vertex_coef;
      int best_vertex_deg = -1;
      int current_vertex_deg;
      EliminationTree::VertexType best_vertex;

      auto g = tree.ComponentsBegin();  // take first component
      for (auto p = std::begin(g->AdjacencyList());
           p != std::end(g->AdjacencyList()); ++p) {
        auto new_components = tree.Eliminate(p->first);
        current_vertex_deg = p->second.size();
        double avg_size =
            std::accumulate(
                std::begin(new_components), std::end(new_components), 0.0,
                [](double acc, auto component) {
                  return acc + component.get().AdjacencyList().size();
                }) /
            new_components.size();

        double gamma = std::accumulate(
            std::begin(new_components), std::end(new_components), 0.0,
            [avg_size](double acc, auto component) {
              return acc +
                     (component.get().AdjacencyList().size() - avg_size) *
                         (component.get().AdjacencyList().size() - avg_size);
            });

        current_vertex_coef =
            D_ / (std::sqrt(gamma + A_)) + E_ * new_components.size();

        if (current_vertex_coef > best_vertex_coef ||
            (current_vertex_coef == best_vertex_coef &&
             current_vertex_deg > best_vertex_deg)) {
          best_vertex_coef = current_vertex_coef;
          best_vertex_deg = current_vertex_deg;
          best_vertex = p->first;
        }

        auto merge_ret = tree.Merge();
        g = merge_ret.first;
        p = merge_ret.second;
      }

      tree.Eliminate(best_vertex);
    }
    return tree.Decompose();
  }

 private:
  std::unique_ptr<BranchAndBound::Heuristic> heuristic_;
  float A_ = 1.0;
  float D_ = 0.5;
  float E_ = 0.5;
};
}  // namespace td
