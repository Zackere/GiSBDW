// Copyright 2020 GISBDW. All rights reserved.
#include "branch_and_bound.hpp"

#include <algorithm>
#include <utility>

namespace td {
namespace {
bool HasChanceImproving(
    EliminationTree::Result const& best_tree,
    EliminationTree::Component const& component,
    std::variant<BranchAndBound::LowerBound::LowerBoundInfo,
                 BranchAndBound::LowerBound::TreedepthInfo> const& v) {
  if (auto* lbinfo =
          std::get_if<BranchAndBound::LowerBound::LowerBoundInfo>(&v)) {
    if (component.Depth() + lbinfo->lower_bound >= best_tree.treedepth)
      return false;
  } else if (auto* tdinfo =
                 std::get_if<BranchAndBound::LowerBound::TreedepthInfo>(&v)) {
    if (component.Depth() + tdinfo->treedepth >= best_tree.treedepth)
      return false;
  }
  return true;
}

std::vector<EliminationTree::VertexType> DefaultAttemptOrder(
    EliminationTree::Component const& component) {
  std::vector<EliminationTree::VertexType> ret;
  ret.reserve(component.AdjacencyList().size());
  for (auto v : component.AdjacencyList())
    ret.emplace_back(v.first);
  return ret;
}
}  // namespace

std::variant<BranchAndBound::LowerBound::LowerBoundInfo,
             BranchAndBound::LowerBound::TreedepthInfo>
BranchAndBound::LowerBound::BetterResult(
    std::variant<LowerBoundInfo, TreedepthInfo>* v1_p,
    std::variant<LowerBoundInfo, TreedepthInfo>* v2_p) {
  if (!v1_p) {
    if (v2_p)
      return *v2_p;
    return LowerBoundInfo{1, std::nullopt};
  }
  if (!v2_p)
    return *v1_p;
  auto& v1 = *v1_p;
  auto& v2 = *v2_p;

  if (auto* td_info1 = std::get_if<TreedepthInfo>(&v1))
    return v1;
  if (auto* td_info2 = std::get_if<TreedepthInfo>(&v2))
    return v2;
  auto& lb_info1 = std::get<LowerBoundInfo>(v1);
  auto& lb_info2 = std::get<LowerBoundInfo>(v2);
  LowerBoundInfo ret{0, std::nullopt};
  ret.lower_bound = std::max(lb_info1.lower_bound, lb_info2.lower_bound);
  if (lb_info1.attempt_order)
    ret.attempt_order = std::move(lb_info1.attempt_order);
  if (lb_info2.attempt_order && ret.attempt_order &&
      lb_info2.attempt_order->size() < ret.attempt_order->size())
    ret.attempt_order = std::move(lb_info2.attempt_order);
  return ret;
}

void BranchAndBound::Algorithm() {
  if (elimination_tree_->ComponentsBegin() ==
      elimination_tree_->ComponentsEnd()) {
    auto result = elimination_tree_->Decompose();
    if (result.treedepth < best_tree_.treedepth)
      best_tree_ = std::move(result);
    return;
  }

  auto begin = elimination_tree_->ComponentsBegin();
  auto& component = *begin;
  auto lb_begin = lower_bound_->Get(*begin);
  if (!HasChanceImproving(best_tree_, component, lb_begin))
    return;
  while (++begin != elimination_tree_->ComponentsEnd())
    if (!HasChanceImproving(best_tree_, *begin, lower_bound_->Get(*begin)))
      return;

  if (auto* tdinfo = std::get_if<LowerBound::TreedepthInfo>(&lb_begin)) {
    for (auto v : tdinfo->elimination_order)
      elimination_tree_->Eliminate(v);
    Algorithm();
    for (int i = 0; i < tdinfo->elimination_order.size(); ++i)
      elimination_tree_->Merge();
    return;
  }
  std::vector<EliminationTree::VertexType> attempt_order;
  auto& lbinfo = std::get<LowerBound::LowerBoundInfo>(lb_begin);
  if (lbinfo.attempt_order)
    attempt_order = std::move(*lbinfo.attempt_order);
  else
    attempt_order = DefaultAttemptOrder(component);
  for (auto v : attempt_order) {
    elimination_tree_->Eliminate(v);
    Algorithm();
    elimination_tree_->Merge();
  }
}
}  // namespace td
