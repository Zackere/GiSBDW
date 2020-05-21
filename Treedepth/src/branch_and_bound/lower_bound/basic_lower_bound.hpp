// Copyright 2020 GISBDW. All rights reserved.
#pragma once

#include "src/branch_and_bound/branch_and_bound.hpp"
#include "src/elimination_tree/elimination_tree.hpp"

namespace td {
class BasicLowerBound : public BranchAndBound::LowerBound {
 public:
  ~BasicLowerBound() override = default;
  std::variant<LowerBoundInfo, TreedepthInfo> Get(
      EliminationTree::Component const& g) override {
    if (g.AdjacencyList().size() > 1)
      return LowerBoundInfo{2, std::nullopt};

    return LowerBoundInfo{1, std::nullopt};
  }
};
}  // namespace td
