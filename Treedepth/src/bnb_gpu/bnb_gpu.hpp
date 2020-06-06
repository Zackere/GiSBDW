// Copyright 2020 GISBDW. All rights reserved.
#pragma once

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/copy.hpp>

namespace td {
class BnBGPU {
 public:
  using BoostGraph =
      boost::adjacency_list<boost::mapS, boost::vecS, boost::undirectedS>;
  template <typename OutEdgeList, typename VertexList, typename... Args>
  void operator()(boost::adjacency_list<OutEdgeList,
                                        VertexList,
                                        boost::undirectedS,
                                        Args...> const& g,
                  std::size_t heur_td);

 private:
  void Run(BoostGraph const& g, std::size_t heur_td);
};
template <typename OutEdgeList, typename VertexList, typename... Args>
inline void BnBGPU::operator()(boost::adjacency_list<OutEdgeList,
                                                     VertexList,
                                                     boost::undirectedS,
                                                     Args...> const& g,
                               std::size_t heur_td) {
  BoostGraph copy;
  boost::copy_graph(g, copy);
  Run(copy, heur_td);
}
}  // namespace td