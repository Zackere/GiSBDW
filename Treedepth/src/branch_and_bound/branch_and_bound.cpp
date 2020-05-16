// Copyright 2020 GISBDW. All rights reserved.
#include "branch_and_bound.hpp"

#include <algorithm>
#include <list>

namespace td {
namespace {
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
  if (lb_info2.attempt_order &&
      (!ret.attempt_order ||
       lb_info2.attempt_order->size() < ret.attempt_order->size()))
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
  auto component_iterator = elimination_tree_->ComponentsBegin();
  auto first_component_lb = lower_bound_->Get(*component_iterator);
  auto first_component_depth = elimination_tree_->ComponentsBegin()->Depth();
  std::list<EliminationTree::VertexType> to_be_eliminated;
  while (++component_iterator != elimination_tree_->ComponentsEnd()) {
    auto lb = lower_bound_->Get(*component_iterator);
    if (auto* lbinfo = std::get_if<LowerBound::LowerBoundInfo>(&lb)) {
      if (component_iterator->Depth() + lbinfo->lower_bound >=
          best_tree_.treedepth)
        return;
    } else if (auto* tdinfo = std::get_if<LowerBound::TreedepthInfo>(&lb)) {
      if (component_iterator->Depth() + tdinfo->treedepth >=
          best_tree_.treedepth)
        return;
      to_be_eliminated.splice(std::end(to_be_eliminated),
                              tdinfo->elimination_order);
    }
  }
  if (auto* lbinfo =
          std::get_if<LowerBound::LowerBoundInfo>(&first_component_lb)) {
    if (first_component_depth + lbinfo->lower_bound >= best_tree_.treedepth)
      return;
    std::vector<EliminationTree::VertexType> attempt_order;
    if (lbinfo->attempt_order)
      attempt_order = std::move(*lbinfo->attempt_order);
    else
      attempt_order =
          DefaultAttemptOrder(*elimination_tree_->ComponentsBegin());
    for (auto v : to_be_eliminated)
      elimination_tree_->Eliminate(v);
    for (auto v : attempt_order) {
      elimination_tree_->Eliminate(v);
      Algorithm();
      elimination_tree_->Merge();
    }
    for (auto v : to_be_eliminated)
      elimination_tree_->Merge();
    return;
  } else if (auto* tdinfo =
                 std::get_if<LowerBound::TreedepthInfo>(&first_component_lb)) {
    if (first_component_depth + tdinfo->treedepth >= best_tree_.treedepth)
      return;
    to_be_eliminated.splice(std::end(to_be_eliminated),
                            tdinfo->elimination_order);
    for (auto v : to_be_eliminated)
      elimination_tree_->Eliminate(v);
    Algorithm();
    for (auto v : to_be_eliminated)
      elimination_tree_->Merge();
    return;
  }
}
}  // namespace td
