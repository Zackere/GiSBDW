// Copyright 2020 GISBDW. All rights reserved.
#include "branch_and_bound.hpp"

namespace td {
BranchAndBound::Heuristic::Heuristic(std::unique_ptr<Heuristic> heuristic)
    : heuristic_(std::move(heuristic)) {}

BranchAndBound::Heuristic* BranchAndBound::Heuristic::Get() {
  return heuristic_.get();
}
}  // namespace td
