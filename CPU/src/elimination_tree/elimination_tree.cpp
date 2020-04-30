// Copyright 2020 GISBDW. All rights reserved.
#include "elimination_tree.hpp"

namespace td {
EliminationTree::Component::AdjacencyList::mapped_type const&
EliminationTree::Component::Neighbours(VertexType v) const {
  return neighbours_.find(v)->second;
}

unsigned EliminationTree::Component::Depth() const {
  return depth_;
}

void EliminationTree::Eliminate(VertexType v) {
  auto& v_node = nodes_[v]->v;
  auto& v_component = std::get<Component>(v_node);
  eliminated_nodes_.emplace_back(v_component.neighbours_.extract(v));
  for (auto& p : v_component.neighbours_)
    p.second.erase(v);
  auto new_v_node = EliminatedNode{std::list<Node>{}, v_component.Depth()};
  while (!v_component.neighbours_.empty()) {
    Node new_node;
    new_node.v = Component();
    auto& new_component = std::get<Component>(new_node.v);
    new_component.depth_ = v_component.Depth() + 1;
    std::map<VertexType, std::set<VertexType>> to_be_added;
    to_be_added.insert(
        v_component.neighbours_.extract(std::begin(v_component.neighbours_)));
    while (!to_be_added.empty()) {
      auto& neigbourhood =
          new_component.neighbours_
              .insert(to_be_added.extract(std::begin(to_be_added)))
              .position->second;
      for (auto neigh : neigbourhood)
        if (auto it = v_component.neighbours_.find(neigh);
            it != std::end(v_component.neighbours_))
          to_be_added.insert(v_component.neighbours_.extract(it));
    }
    new_v_node.children.push_back(std::move(new_node));
    auto& back_ref = std::get<Component>(new_v_node.children.back().v);
    for (auto& p : back_ref.neighbours_)
      nodes_[p.first] = &new_v_node.children.back();
    components_.insert(&back_ref);
  }
  components_.erase(&v_component);
  v_node = std::move(new_v_node);
}

void EliminationTree::Merge() {}

EliminationTree::ComponentIterator EliminationTree::ComponentsBegin() const {
  return ComponentIterator{std::cbegin(components_)};
}

EliminationTree::ComponentIterator EliminationTree::ComponentsEnd() const {
  return ComponentIterator{std::cend(components_)};
}

EliminationTree::Component const&
EliminationTree::ComponentIterator::operator*() const {
  return **current_;
}

EliminationTree::Component const*
EliminationTree::ComponentIterator::operator->() const {
  return *current_;
}

EliminationTree::ComponentIterator&
EliminationTree::ComponentIterator::operator++() {
  ++current_;
  return *this;
}

EliminationTree::ComponentIterator&
EliminationTree::ComponentIterator::operator--() {
  --current_;
  return *this;
}

bool EliminationTree::ComponentIterator::operator==(
    ComponentIterator const& other) {
  return current_ == other.current_;
}

bool EliminationTree::ComponentIterator::operator!=(
    ComponentIterator const& other) {
  return current_ != other.current_;
}

EliminationTree::ComponentIterator::ComponentIterator(Iterator init)
    : current_{init} {}
}  // namespace td
