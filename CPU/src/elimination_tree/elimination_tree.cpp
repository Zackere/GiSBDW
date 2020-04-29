#include "elimination_tree.hpp"

namespace td {
EliminationTree::Component::AdjacencyList::mapped_type const&
EliminationTree::Component::Neighbours(VertexType v) const {
  return neighbours_.find(v)->second;
}

unsigned EliminationTree::Component::GetDepth() const {
  return depth_;
}

void EliminationTree::Eliminate(VertexType v) {}

void EliminationTree::Merge(VertexType v) {}

EliminationTree::ComponentIterator EliminationTree::ComponentsBegin() const {
  return ComponentIterator{std::cbegin(components_)};
}

EliminationTree::ComponentIterator EliminationTree::ComponentsEnd() const {
  return ComponentIterator{std::cend(components_)};
}

EliminationTree::Component const&
EliminationTree::ComponentIterator::operator*() const {
  return cur_->second;
}

EliminationTree::Component const*
EliminationTree::ComponentIterator::operator->() const {
  return &cur_->second;
}

EliminationTree::ComponentIterator&
EliminationTree::ComponentIterator::operator++() {
  ++cur_;
  return *this;
}

EliminationTree::ComponentIterator&
EliminationTree::ComponentIterator::operator--() {
  --cur_;
  return *this;
}

bool EliminationTree::ComponentIterator::operator==(
    ComponentIterator const& other) {
  return cur_ == other.cur_;
}

bool EliminationTree::ComponentIterator::operator!=(
    ComponentIterator const& other) {
  return cur_ != other.cur_;
}

EliminationTree::ComponentIterator::ComponentIterator(Iterator init)
    : cur_{init} {}
}  // namespace td
