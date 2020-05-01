// Copyright 2020 GISBDW. All rights reserved.
#include "branch_and_bound.hpp"

#include <algorithm>
#include <utility>

namespace td {
void BranchAndBound::Algorithm() {
  for (auto it = elimination_tree_->ComponentsBegin();
       it != elimination_tree_->ComponentsEnd(); ++it)
    if (lower_bound_->Get(*it) + it->Depth() >= best_tree_.treedepth)
      return;

  if (free_vertices_.empty()) {
    auto result = elimination_tree_->Decompose();
    if (result.treedepth < best_tree_.treedepth)
      best_tree_ = std::move(result);
    return;
  }

  for (auto iter_v = std::begin(free_vertices_);
       iter_v != std::end(free_vertices_); ++iter_v) {
    elimination_tree_->Eliminate(*iter_v);
    free_vertices_.erase(iter_v);
    Algorithm();
    iter_v = free_vertices_.insert(elimination_tree_->Merge()).first;
  }
}
}  // namespace td
