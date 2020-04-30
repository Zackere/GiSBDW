// Copyright 2020 GISBDW. All rights reserved.
#include "elimination_tree.hpp"

namespace td {
EliminationTree::Component::AdjacencyListType const&
EliminationTree::Component::AdjacencyList() const {
  return neighbours_;
}

unsigned EliminationTree::Component::Depth() const {
  return depth_;
}

bool EliminationTree::Component::operator==(Component const& other) const {
  return depth_ == other.depth_ && neighbours_ == other.neighbours_;
}

void EliminationTree::Eliminate(VertexType v) {
  auto& v_node = nodes_[v]->v;
  auto& v_component = std::get<Component>(v_node);
  // Remove v from graph and save its adjacency list for later
  eliminated_nodes_.emplace_back(v_component.neighbours_.extract(v));
  for (auto& p : v_component.neighbours_)
    p.second.erase(v);
  auto new_v_node = EliminatedNode{std::list<Node>{}, v, v_component.Depth()};
  while (!v_component.neighbours_.empty()) {
    Node new_node;
    new_node.v = Component();
    auto& new_component = std::get<Component>(new_node.v);
    new_component.depth_ = v_component.Depth() + 1;
    // Desctructive BFS (separate connected components of v_component into
    // separate Component objects)
    decltype(v_component.neighbours_) to_be_added;
    to_be_added.insert(
        v_component.neighbours_.extract(std::begin(v_component.neighbours_)));
    while (!to_be_added.empty()) {
      // Extract vertex to currently built component
      auto& neigbourhood =
          new_component.neighbours_
              .insert(to_be_added.extract(std::begin(to_be_added)))
              .position->second;
      // Schedule neighbourhood to be added into currently built component
      for (auto neigh : neigbourhood)
        if (auto it = v_component.neighbours_.find(neigh);
            it != std::end(v_component.neighbours_))
          to_be_added.insert(v_component.neighbours_.extract(it));
    }
    new_v_node.children.push_back(std::move(new_node));
    // Update location of vertices inside new_node
    auto& back_ref = std::get<Component>(new_v_node.children.back().v);
    for (auto& p : back_ref.neighbours_)
      nodes_[p.first] = &new_v_node.children.back();
    components_.insert(&back_ref);
  }
  components_.erase(&v_component);
  v_node = std::move(new_v_node);
}

void EliminationTree::Merge() {
  Component new_component;
  // Insert last removed vertex into new_component
  auto to_be_merged =
      new_component.neighbours_.insert(std::move(eliminated_nodes_.back()))
          .position;
  eliminated_nodes_.pop_back();
  auto& node = nodes_[to_be_merged->first];
  new_component.depth_ = std::get<EliminatedNode>(node->v).depth;
  // Merge child components into single graph
  for (auto& child_component : std::get<EliminatedNode>(node->v).children) {
    for (auto& p : std::get<Component>(child_component.v).neighbours_)
      new_component.neighbours_[p.first] = std::move(p.second);
    components_.erase(&std::get<Component>(child_component.v));
  }
  // Add edges which were lost upon elimination
  for (auto& p : to_be_merged->second)
    new_component.neighbours_[p].insert(to_be_merged->first);
  node->v = std::move(new_component);
  // Update location of vertices inside new_component
  auto& merged_component = std::get<Component>(node->v);
  for (auto& p : merged_component.neighbours_)
    nodes_[p.first] = node;
  components_.insert(&merged_component);
}

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
