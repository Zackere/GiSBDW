// Copyright 2020 GISBDW. All rights reserved.
#pragma once

#include <list>
#include <map>
#include <set>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>
#ifdef TD_CHECK_ARGS
#include <stdexcept>
#endif

#include "boost/graph/adjacency_list.hpp"
#include "boost/graph/connected_components.hpp"

class EliminationTreeFixture;
namespace td {
class EliminationTree {
 public:
  using VertexType = std::size_t;
  using ComponentIndex = std::size_t;
  class Component {
   public:
    using AdjacencyList = std::map<VertexType, std::set<VertexType>>;
    AdjacencyList::mapped_type const& Neighbours(VertexType v) const;
    unsigned Depth() const;

   private:
    friend class EliminationTree;
    friend class EliminationTreeFixture;
    AdjacencyList neighbours_;
    unsigned depth_;
  };
  class ComponentIterator {
   public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = Component;
    using difference_type = size_t;
    using pointer = Component*;
    using reference = Component&;

    Component const& operator*() const;
    Component const* operator->() const;

    ComponentIterator& operator++();
    ComponentIterator& operator--();

    bool operator==(ComponentIterator const& other) const;
    bool operator!=(ComponentIterator const& other) const;

   private:
    friend class EliminationTree;
    using Iterator = std::set<Component*>::const_iterator;
    explicit ComponentIterator(Iterator init);
    Iterator current_;
  };

  template <typename OutEdgeList, typename VertexList, typename... Args>
  explicit EliminationTree(boost::adjacency_list<OutEdgeList,
                                                 VertexList,
                                                 boost::undirectedS,
                                                 Args...> const& g);
  void Eliminate(VertexType v);
  void Merge();

  ComponentIterator ComponentsBegin() const;
  ComponentIterator ComponentsEnd() const;

 private:
  struct Node;
  struct EliminatedNode {
    std::list<Node> children;
    unsigned depth;
  };
  struct Node {
    std::variant<EliminatedNode, Component> v;
  } root_;
  std::vector<Node*> nodes_;
  std::set<Component*> components_;
  std::vector<Component::AdjacencyList::node_type> eliminated_nodes_;

  EliminationTree(EliminationTree const&) = delete;
  EliminationTree& operator=(EliminationTree const&) = delete;
};

template <typename OutEdgeList, typename VertexList, typename... Args>
EliminationTree::EliminationTree(boost::adjacency_list<OutEdgeList,
                                                       VertexList,
                                                       boost::undirectedS,
                                                       Args...> const& g)
    : nodes_(boost::num_vertices(g), &root_) {
#ifdef TD_CHECK_ARGS
  if (boost::connected_components(
          g, std::vector<int>(boost::num_vertices(g)).data()) != 1)
    throw std::invalid_argument(
        "EliminationTree works only on connected graphs");
  for (int i = 0; i < boost::num_vertices(g); ++i)
    if (boost::edge(i, i, g).second)
      throw std::invalid_argument("Self loops are not allowed");
#endif
  eliminated_nodes_.reserve(boost::num_vertices(g));
  Component root;
  root.depth_ = 0;
  typename boost::graph_traits<
      std::remove_reference_t<decltype(g)>>::out_edge_iterator ei,
      ei_end;
  for (int i = 0; i < boost::num_vertices(g); ++i)
    for (boost::tie(ei, ei_end) = out_edges(i, g); ei != ei_end; ++ei)
      root.neighbours_[i].insert(target(*ei, g));
  root_.v = std::move(root);
  components_.insert(&std::get<Component>(root_.v));
}
}  // namespace td
