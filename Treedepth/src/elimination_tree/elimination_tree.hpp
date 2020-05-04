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

class ParametrizedEliminationTreeFixture;
namespace td {
/* Class representing treedepth decomposition of incomplete elimination defined
 * in introductory documentation.
 */
class EliminationTree {
 public:
  using VertexType = std::size_t;
  using BoostGraph =
      boost::adjacency_list<boost::mapS, boost::vecS, boost::undirectedS>;
  struct Result {
    BoostGraph td_decomp;
    unsigned treedepth;
    VertexType root;
  };
  /**
   * Class representing a connected component of a graph after some
   * elimination.
   */
  class Component {
   public:
    using AdjacencyListType = std::map<VertexType, std::set<VertexType>>;
    /* *
     * Access adjacency list defined by component.
     *
     * @return Adajcency list of Component.
     */
    AdjacencyListType const& AdjacencyList() const;
    /**
     * @return Depth of component inside EliminationTree that owns it.
     */
    unsigned Depth() const;

    /**
     * Checks if objects represent the same graph on the same depth.
     *
     * @return true if objects are the same. false otherwise.
     */
    bool operator==(Component const& other) const;

   private:
    friend class EliminationTree;
    friend class ::ParametrizedEliminationTreeFixture;
    AdjacencyListType neighbours_;
    unsigned depth_;
  };
  /**
   * Class used to iterate over leaves in EliminationTree which are represented
   * by a graph.
   */
  class ComponentIterator {
   public:
    using iterator_category = std::bidirectional_iterator_tag;
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

  /**
   * Initiates EliminationTree object to T_eps.
   *
   * @param g connected and without self-loops undirected graph which will
   * become a root of T_eps
   */
  template <typename OutEdgeList, typename VertexList, typename... Args>
  explicit EliminationTree(boost::adjacency_list<OutEdgeList,
                                                 VertexList,
                                                 boost::undirectedS,
                                                 Args...> const& g);
  /**
   * Performs elimination of uneliminated vertex v. This operation is analogous
   * to transformation T_w->T_wv
   *
   * @param v vertex to be eliminated
   */
  void Eliminate(VertexType v);
  /**
   * Reverts last recorded elimination. This operation is analogous
   * to transformation T_wv->T_w
   */
  VertexType Merge();

  /**
   *  @return Iterator to the first Component of EliminationTree.
   */
  ComponentIterator ComponentsBegin() const;
  /**
   * @return Iterator to the Component following the last Component of
   * EliminationTree.
   */
  ComponentIterator ComponentsEnd() const;
  /**
   * Converts EliminationTree to BoostGraph. Works only when all vertices had
   * been eliminated.
   *
   * Usage: auto [g, depth, root] = et.Decompose();
   *
   * @return EliminationTree representation in BoostGraph along with its
   * treedepth and root.
   */
  Result Decompose() const;

 private:
  struct Node;
  struct EliminatedNode {
    std::list<Node> children;
    VertexType vertex;
    unsigned depth;
  };
  struct Node {
    std::variant<EliminatedNode, Component> v;
  } root_;
  std::vector<Node*> nodes_;
  std::set<Component*> components_;
  std::vector<Component::AdjacencyListType::node_type> eliminated_nodes_;

  EliminationTree(EliminationTree const&) = delete;
  EliminationTree& operator=(EliminationTree const&) = delete;
};

template <typename OutEdgeList, typename VertexList, typename... Args>
inline EliminationTree::EliminationTree(
    boost::adjacency_list<OutEdgeList,
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
    for (boost::tie(ei, ei_end) = boost::out_edges(i, g); ei != ei_end; ++ei)
      root.neighbours_[i].insert(boost::target(*ei, g));
  root_.v = std::move(root);
  components_.insert(&std::get<Component>(root_.v));
}

inline EliminationTree::Result EliminationTree::Decompose() const {
#ifdef TD_CHECK_ARGS
  if (eliminated_nodes_.size() != nodes_.size())
    throw std::runtime_error("Elimination is not complete");
#endif
  Result ret = {BoostGraph(nodes_.size()), 0,
                std::get<EliminatedNode>(root_.v).vertex};
  std::set<VertexType> insert_now, insert_next;
  insert_now.insert(std::get<EliminatedNode>(root_.v).vertex);
  while (!insert_now.empty()) {
    for (auto v : insert_now) {
      auto& v_node = std::get<EliminatedNode>(nodes_[v]->v);
      for (auto& p : v_node.children) {
        auto& p_node = std::get<EliminatedNode>(p.v);
        boost::add_edge(v, p_node.vertex, ret.td_decomp);
        insert_next.insert(p_node.vertex);
      }
    }
    insert_now = std::move(insert_next);
    ++ret.treedepth;
  }
  return ret;
}
}  // namespace td
