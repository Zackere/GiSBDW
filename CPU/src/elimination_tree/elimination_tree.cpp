#include "elimination_tree.hpp"

namespace td {
std::set<EliminationTree::VertexType> const&
EliminationTree::Component::Neighbours(VertexType v) const {
  return neighbours_[v];
}

void EliminationTree::Eliminate(VertexType v) {}

void EliminationTree::Merge(VertexType v) {}

EliminationTree::Component const& EliminationTree::GetComponent(
    VertexType v) const {
  return components_[nodes_[v]];
}
}  // namespace td
