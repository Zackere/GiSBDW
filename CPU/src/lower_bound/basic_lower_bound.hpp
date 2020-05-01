// Copyright 2020 GISBDW. All rights reserved.
#pragma once

#include "../branch_and_bound/branch_and_bound.hpp"
#include "../elimination_tree/elimination_tree.hpp"

namespace td {

class BasicLowerBound : public BranchAndBound::LowerBound {
 public:
  ~BasicLowerBound() override = default;
  unsigned Get(EliminationTree::Component const& g) override {
    if (g.AdjacencyList().size() > 1)
      return 2;

    return 1;
  }
};
}  // namespace td
