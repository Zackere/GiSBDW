// Copyright 2020 GISBDW. All rights reserved.
#pragma once

#include <memory>
#include <utility>

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
  };
  class Heuristic {
    // dekorator lista Heurystyka(Heurystyka(...));
   public:
    struct Result {
      Graph td_decomp;
      unsigned depth;
      unsigned root;
    };
    virtual Result Get(Graph const& g) = 0;
    std::unique_ptr<Heuristic> heuristic;
  };
  template <typename OutEdgeList, typename VertexList, typename... Args>
  Graph Run(boost::adjacency_list<OutEdgeList,
                                  VertexList,
                                  boost::undirectedS,
                                  Args...> const& g,
            std::unique_ptr<LowerBound> lower_bound,
            std::unique_ptr<Heuristic> heuristic);

 private:
  std::unique_ptr<EliminationTree> elimination_tree_ = nullptr;
  std::unique_ptr<LowerBound> lower_bound_ = nullptr;
};

template <typename OutEdgeList, typename VertexList, typename... Args>
inline typename BranchAndBound::Graph BranchAndBound::Run(
    boost::adjacency_list<OutEdgeList,
                          VertexList,
                          boost::undirectedS,
                          Args...> const& g,
    std::unique_ptr<LowerBound> lower_bound,
    std::unique_ptr<Heuristic> heuristic) {
#ifdef TD_CHECK_ARGS
  // check if g is connected
#endif
  // Graph graph = g; wont work
  // auto res = heuristic->Get(graph);
  return Graph();
}
}  // namespace td
