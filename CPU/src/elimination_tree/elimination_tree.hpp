// Copyright 2020 GISBDW. All rights reserved.
#pragma once

#include <map>
#include <set>
#ifdef TD_CHECK_ARGS
#include <stdexcept>
#endif
#include <variant>
#include <vector>

#include "boost/graph/adjacency_list.hpp"
#include "boost/graph/connected_components.hpp"

namespace td {
class EliminationTree {
 public:
  using VertexType = std::size_t;
  using ComponentIndex = std::size_t;
  class Component {
   public:
    using AdjacencyList = std::map<VertexType, std::set<VertexType>>;
    AdjacencyList::mapped_type const& Neighbours(VertexType v) const;
    unsigned GetDepth() const;

   private:
    AdjacencyList neighbours_;
    unsigned depth_;
  };
  class ComponentIterator {
   public:
    ComponentIterator(ComponentIterator const&) = default;
    ComponentIterator(ComponentIterator&&) = default;
    ComponentIterator& operator=(ComponentIterator const&) = default;
    ComponentIterator& operator=(ComponentIterator&&) = default;
    ~ComponentIterator() = default;

    Component const& operator*() const;
    Component const* operator->() const;

    ComponentIterator& operator++();
    ComponentIterator& operator--();

    bool operator==(ComponentIterator const& other);
    bool operator!=(ComponentIterator const& other);

   private:
    friend class EliminationTree;
    using Iterator = std::map<ComponentIndex, Component>::const_iterator;
    ComponentIterator(Iterator init);
    Iterator cur_;
  };

  template <typename OutEdgeList, typename VertexList, typename... Args>
  EliminationTree(boost::adjacency_list<OutEdgeList,
                                        VertexList,
                                        boost::undirectedS,
                                        Args...> const& g);
  void Eliminate(VertexType v);
  void Merge(VertexType v);

  ComponentIterator ComponentsBegin() const;
  ComponentIterator ComponentsEnd() const;

 private:
  struct Node {
    std::variant<std::pair<std::vector<Node>, unsigned>, ComponentIndex>
        children;
  };
  std::vector<Node> nodes_;
  std::map<ComponentIndex, Component> components_;
  std::vector<Component::AdjacencyList::node_type> eliminated_;
};

template <typename OutEdgeList, typename VertexList, typename... Args>
EliminationTree::EliminationTree(boost::adjacency_list<OutEdgeList,
                                                       VertexList,
                                                       boost::undirectedS,
                                                       Args...> const& g)
    : nodes_(boost::num_vertices(g)) {
#ifdef TD_CHECK_ARGS
  if (boost::connected_components(
          g, std::vector<int>(boost::num_vertices(g)).data()) != 1)
    throw std::invalid_argument(
        "EliminationTree works only on connected graphs");
#endif
  eliminated_.reserve(boost::num_vertices(g));
}
}  // namespace td
