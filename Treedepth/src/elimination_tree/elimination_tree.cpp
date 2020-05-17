// Copyright 2020 GISBDW. All rights reserved.
#include "elimination_tree.hpp"

#include <string>

namespace td {
EliminationTree::Component::AdjacencyListType const&
EliminationTree::Component::AdjacencyList() const {
  return neighbours_;
}

unsigned EliminationTree::Component::Depth() const {
  return depth_;
}

unsigned EliminationTree::Component::NEdges() const {
  return nedges_;
}

bool EliminationTree::Component::operator==(Component const& other) const {
  return depth_ == other.depth_ && neighbours_ == other.neighbours_;
}

std::list<std::set<EliminationTree::Component,
                   EliminationTree::ComponentCmp>::const_iterator>
EliminationTree::Eliminate(VertexType v) {
#ifdef TD_CHECK_ARGS
  if (!std::get_if<decltype(components_)::const_iterator>(&nodes_[v].get().v))
    throw std::invalid_argument("Vertex " + std::to_string(v) +
                                " is eliminated");
#endif
  std::list<decltype(components_)::const_iterator> ret;
  auto node = nodes_[v];
  auto component = components_.extract(
      std::get<decltype(components_)::const_iterator>(nodes_[v].get().v));
  node.get().v = EliminatedNode{{}, v, component.value().Depth()};
  eliminated_nodes_.emplace_back(component.value().neighbours_.extract(v));
  for (auto& p : component.value().neighbours_)
    p.second.erase(v);
  while (!component.value().AdjacencyList().empty()) {
    Component new_component;
    new_component.depth_ = component.value().Depth() + 1;
    new_component.nedges_ = 0;
    decltype(component.value().neighbours_) to_be_added;
    to_be_added.insert(component.value().neighbours_.extract(
        std::begin(component.value().neighbours_)));
    while (!to_be_added.empty()) {
      // Extract vertex to currently built component
      auto& neigbourhood =
          new_component.neighbours_
              .insert(to_be_added.extract(std::begin(to_be_added)))
              .position->second;

      new_component.nedges_ += neigbourhood.size();
      // Schedule neighbourhood to be added into currently built component
      for (auto neigh : neigbourhood)
        if (auto it = component.value().neighbours_.find(neigh);
            it != std::end(component.value().neighbours_))
          to_be_added.insert(component.value().neighbours_.extract(it));
    }
    new_component.nedges_ /= 2;

    auto new_component_pos = components_.insert(std::move(new_component)).first;
    auto& node_ref = std::get<EliminatedNode>(node.get().v)
                         .children.emplace_back(Node{new_component_pos});
    for (auto& p : new_component_pos->AdjacencyList())
      nodes_[p.first] = node_ref;
    ret.push_back(new_component_pos);
  }
  return ret;
}

std::pair<EliminationTree::ComponentIterator,
          EliminationTree::Component::AdjacencyListType::const_iterator>
EliminationTree::Merge() {
#ifdef TD_CHECK_ARGS
  if (eliminated_nodes_.size() == 0)
    throw std::runtime_error("No vertex to merge");
#endif
  Component new_component;
  auto vertex_being_merged =
      new_component.neighbours_.insert(std::move(eliminated_nodes_.back()))
          .position;
  eliminated_nodes_.pop_back();
  for (auto& child_component_node :
       std::get<EliminatedNode>(nodes_[vertex_being_merged->first].get().v)
           .children) {
    auto child_component =
        components_.extract(std::get<decltype(components_)::const_iterator>(
            child_component_node.v));
    for (auto& p : child_component.value().neighbours_) {
      new_component.nedges_ += p.second.size();
      new_component.neighbours_[p.first] = std::move(p.second);
    }
  }
  for (auto v : vertex_being_merged->second)
    new_component.neighbours_[v].insert(vertex_being_merged->first);
  new_component.depth_ =
      std::get<EliminatedNode>(nodes_[vertex_being_merged->first].get().v)
          .depth;
  new_component.nedges_ += vertex_being_merged->second.size();
  new_component.nedges_ /= 2;

  auto new_component_pos = components_.insert(std::move(new_component)).first;
  nodes_[vertex_being_merged->first].get().v = new_component_pos;
  for (auto& p : new_component_pos->AdjacencyList())
    nodes_[p.first] = nodes_[vertex_being_merged->first];
  return {ComponentIterator{new_component_pos}, vertex_being_merged};
}

EliminationTree::ComponentIterator EliminationTree::ComponentsBegin() const {
  return ComponentIterator{std::cbegin(components_)};
}

EliminationTree::ComponentIterator EliminationTree::ComponentsEnd() const {
  return ComponentIterator{std::cend(components_)};
}

EliminationTree::Component const&
EliminationTree::ComponentIterator::operator*() const {
  return *current_;
}

EliminationTree::Component const*
EliminationTree::ComponentIterator::operator->() const {
  return &*current_;
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
    ComponentIterator const& other) const {
  return current_ == other.current_;
}

bool EliminationTree::ComponentIterator::operator!=(
    ComponentIterator const& other) const {
  return current_ != other.current_;
}

EliminationTree::ComponentIterator::ComponentIterator(Iterator init)
    : current_{init} {}
}  // namespace td
