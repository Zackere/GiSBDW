// Copyright 2020 GISBDW. All rights reserved.
#pragma once

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/copy.hpp>
#include <map>
#include <tuple>
#include <vector>

#include "../elimination_tree/elimination_tree.hpp"
#include "../set_encoder/set_encoder.hpp"

namespace td {
class DynamicCPU {
 public:
  using BoostGraph =
      boost::adjacency_list<boost::mapS, boost::vecS, boost::undirectedS>;

  template <typename OutEdgeList, typename VertexList, typename... Args>
  void operator()(boost::adjacency_list<OutEdgeList,
                                        VertexList,
                                        boost::undirectedS,
                                        Args...> const& g);

  std::size_t GetIterationsPerformed() const;

  template <typename OutEdgeList, typename VertexList, typename... Args>
  EliminationTree::Result GetTDDecomp(
      std::size_t code,
      boost::adjacency_list<OutEdgeList,
                            VertexList,
                            boost::undirectedS,
                            Args...> const& induced_graph);

  std::size_t GetTreedepth(std::size_t nverts,
                           std::size_t subset_size,
                           std::size_t subset_code);

 private:
  void Run(BoostGraph const& g);

  std::vector<std::map<std::size_t, std::tuple<std::size_t, int>>> history_;
};
template <typename OutEdgeList, typename VertexList, typename... Args>
inline void DynamicCPU::operator()(boost::adjacency_list<OutEdgeList,
                                                         VertexList,
                                                         boost::undirectedS,
                                                         Args...> const& g) {
  BoostGraph graph;
  boost::copy_graph(g, graph);
  Run(graph);
}

template <typename OutEdgeList, typename VertexList, typename... Args>
inline EliminationTree::Result DynamicCPU::GetTDDecomp(
    std::size_t code,
    boost::adjacency_list<OutEdgeList,
                          VertexList,
                          boost::undirectedS,
                          Args...> const& induced_graph) {
  if (boost::num_vertices(induced_graph) < history_.size())
    if (auto it = history_[boost::num_vertices(induced_graph)].find(code);
        it != std::end(history_[boost::num_vertices(induced_graph)])) {
      td::EliminationTree et(induced_graph);
      while (et.ComponentsBegin() != et.ComponentsEnd())
        et.Eliminate(
            std::get<1>(history_[et.ComponentsBegin()->AdjacencyList().size()]
                                [set_encoder::Encode(
                                    et.ComponentsBegin()->AdjacencyList())]));
      return et.Decompose();
    }
  return EliminationTree::Result{EliminationTree::BoostGraph(),
                                 std::numeric_limits<unsigned>::max(), 0};
}
}  // namespace td
