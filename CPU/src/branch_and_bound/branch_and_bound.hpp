// Copyright 2020 GISBDW. All rights reserved.
#pragma once

#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "../elimination_tree/elimination_tree.hpp"
#include "boost/graph/adjacency_list.hpp"

namespace td {
class BranchAndBound {
 public:
  using Graph =
      boost::adjacency_list<boost::mapS, boost::vecS, boost::undirectedS>;
  class LowerBound {
   public:
    virtual unsigned Get(EliminationTree::Component const& g) { return 1; }
    virtual ~LowerBound() = default;
  };
  class Heuristic {
   public:
    explicit Heuristic(std::unique_ptr<Heuristic> heuristic);
    virtual EliminationTree::Result Get(Graph const& g) = 0;
    virtual ~Heuristic() = default;

   protected:
    Heuristic* Get();

   private:
    std::unique_ptr<Heuristic> heuristic_;
  };
  template <typename OutEdgeList, typename VertexList, typename... Args>
  EliminationTree::Result operator()(boost::adjacency_list<OutEdgeList,
                                                           VertexList,
                                                           boost::undirectedS,
                                                           Args...> const& g,
                                     std::unique_ptr<LowerBound> lower_bound,
                                     std::unique_ptr<Heuristic> heuristic);

 private:
  void Algorithm();
  EliminationTree::Result best_tree_;
  std::set<EliminationTree::VertexType> free_vertices_;

  std::unique_ptr<EliminationTree> elimination_tree_ = nullptr;
  std::unique_ptr<LowerBound> lower_bound_ = nullptr;
};

template <typename OutEdgeList, typename VertexList, typename... Args>
inline typename EliminationTree::Result BranchAndBound::operator()(
    boost::adjacency_list<OutEdgeList,
                          VertexList,
                          boost::undirectedS,
                          Args...> const& g,
    std::unique_ptr<LowerBound> lower_bound,
    std::unique_ptr<Heuristic> heuristic) {
#ifdef TD_CHECK_ARGS
  if (boost::connected_components(
          g, std::vector<int>(boost::num_vertices(g)).data()) != 1)
    throw std::invalid_argument(
        "EliminationTree works only on connected graphs");
  for (int i = 0; i < boost::num_vertices(g); ++i)
    if (boost::edge(i, i, g).second)
      throw std::invalid_argument("Self loops are not allowed");
#endif
  elimination_tree_ = std::make_unique<EliminationTree>(g);
  lower_bound_ = std::move(lower_bound);
  free_vertices_.clear();
  for (int i = 0; i < boost::num_vertices(g); ++i)
    free_vertices_.insert(i);
  best_tree_ = heuristic->Get(g);
  Algorithm();
  return best_tree_;
}
}  // namespace td
