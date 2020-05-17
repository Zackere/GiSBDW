// Copyright 2020 GISBDW. All rights reserved.
#pragma once

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <list>
#include <map>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

class ParametrizedEliminationTreeFixture;
namespace td {
/* *
 * Class representing treedepth decomposition of incomplete elimination defined
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
     * @return Number of edges inside a component.
     */
    unsigned NEdges() const;

    /**
     * Checks if objects represent the same graph on the same depth.
     *
     * @return true if objects are the same. false otherwise.
     */
    bool operator==(Component const& other) const;

   private:
    friend class EliminationTree;
    friend class ::ParametrizedEliminationTreeFixture;
    AdjacencyListType neighbours_ = {};
    unsigned depth_ = 0;
    unsigned nedges_ = 0;
  };
  struct ComponentCmp {
    bool operator()(Component const& c1, Component const& c2) const {
      return std::begin(c1.AdjacencyList())->first <
             std::begin(c2.AdjacencyList())->first;
    }
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
    using Iterator = std::set<Component, ComponentCmp>::const_iterator;
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
  std::list<std::set<EliminationTree::Component,
                     EliminationTree::ComponentCmp>::const_iterator>
  Eliminate(VertexType v);
  /**
   * Reverts last recorded elimination. This operation is analogous
   * to transformation T_wv->T_w
   */
  std::pair<ComponentIterator, Component::AdjacencyListType::const_iterator>
  Merge();

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
   * @return EliminationTree representation in BoostGraph along with its
   * treedepth and root.
   */
  Result Decompose() const;

 private:
  struct Node;
  struct EliminatedNode {
    std::list<Node> children = {};
    VertexType vertex = 0;
    unsigned depth = 0;
  };
  struct Node {
    std::variant<EliminatedNode,
                 std::set<Component, ComponentCmp>::const_iterator>
        v = EliminatedNode{};
  } root_;
  std::vector<std::reference_wrapper<Node>> nodes_;
  std::set<Component, ComponentCmp> components_;
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
    : nodes_(boost::num_vertices(g), root_) {
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
  root.nedges_ = boost::num_edges(g);
  typename boost::graph_traits<
      std::remove_reference_t<decltype(g)>>::out_edge_iterator ei,
      ei_end;
  for (int i = 0; i < boost::num_vertices(g); ++i)
    for (boost::tie(ei, ei_end) = boost::out_edges(i, g); ei != ei_end; ++ei)
      root.neighbours_[i].insert(boost::target(*ei, g));
  root_.v = components_.insert(std::move(root)).first;
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
    for (auto const v : insert_now) {
      auto const& v_node = std::get<EliminatedNode>(nodes_[v].get().v);
      for (auto const& p : v_node.children) {
        auto const& p_node = std::get<EliminatedNode>(p.v);
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
