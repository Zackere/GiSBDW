#include "elimination_tree.hpp"

namespace td {
std::set<EliminationTree::VertexType> const&
EliminationTree::Component::Neighbours(VertexType v) const {
  return neighbours_.find(v)->second;
}

void EliminationTree::Eliminate(VertexType v) {}

void EliminationTree::Merge(VertexType v) {}

EliminationTree::Component const& EliminationTree::GetComponent(
    VertexType v) const {
  return components_.find(std::get<ComponentIndex>(nodes_[v].children))->second;
}
}  // namespace td
