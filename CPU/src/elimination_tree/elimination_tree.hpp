// Copyright 2020 GISBDW. All rights reserved.
#pragma once

#include <map>
#include <set>
#include <variant>
#include <vector>

#include "boost/graph/adjacency_list.hpp"

namespace td {
class EliminationTree {
 public:
  using Graph =
      boost::adjacency_list<boost::mapS, boost::vecS, boost::undirectedS>;
  using VertexType = int;
  using ComponentIndex = int;
  class Component {
    std::set<VertexType> const& Neighbours(VertexType v) const;

   private:
    std::map<VertexType, std::set<VertexType>> neighbours_;
  };

  template <typename G>
  EliminationTree(G const& g);

  void Eliminate(VertexType v);
  void Merge(VertexType v);
  Component const& GetComponent(VertexType v) const;
  // TODO(replinw): implement possibility of iterating over component

 private:
  struct Node {
    std::variant<std::vector<Node>, ComponentIndex> children;
  };
  VertexType root_;
  std::vector<Node> nodes_;
  std::map<ComponentIndex, Component> components_;
  Graph original_graph_;
  std::vector<VertexType> eliminated_;
};

template <typename G>
EliminationTree::EliminationTree(G const& g) {}
}  // namespace td
