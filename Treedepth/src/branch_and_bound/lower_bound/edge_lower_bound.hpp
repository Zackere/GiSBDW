// Copyright 2020 GISBDW. All rights reserved.
#pragma once

#include <cmath>

#include "src/branch_and_bound/branch_and_bound.hpp"
#include "src/elimination_tree/elimination_tree.hpp"

namespace td {
class EdgeLowerBound : public BranchAndBound::LowerBound {
 public:
  ~EdgeLowerBound() override = default;
  std::variant<LowerBoundInfo, TreedepthInfo> Get(
      EliminationTree::Component const& g) override {
    int e = g.NEdges();
    int n = g.AdjacencyList().size();
    return LowerBoundInfo{static_cast<unsigned>(std::ceil(
                              0.5 + n - std::sqrt(0.25 + n * n - n - 2 * e))),
                          std::nullopt};
  }
};
}  // namespace td
