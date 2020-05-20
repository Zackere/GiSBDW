// Copyright 2020 GISBDW. All rights reserved.
#pragma once

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/copy.hpp>
#include <map>
#include <memory>
#include <tuple>
#include <vector>

#include "../elimination_tree/elimination_tree.hpp"
#include "../set_encoder/set_encoder.hpp"

namespace td {
class DynamicCPUImprov {
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
  std::size_t Run(EliminationTree::ComponentIterator component);
  EliminationTree::Result GetTDDecompImpl(std::size_t code,
                                          BoostGraph const& g);

  std::unique_ptr<EliminationTree> et_;
  std::vector<std::unordered_map<std::size_t, std::tuple<std::size_t, int>>>
      history_;
};
template <typename OutEdgeList, typename VertexList, typename... Args>
inline void DynamicCPUImprov::operator()(
    boost::adjacency_list<OutEdgeList,
                          VertexList,
                          boost::undirectedS,
                          Args...> const& g) {
  et_.reset(new td::EliminationTree(g));
  history_.reserve(0);
  history_.resize(boost::num_vertices(g) + 1);
  history_[0][0] = {0, -1};
  for (std::size_t i = 0; i < boost::num_vertices(g); ++i)
    history_[1][1 << i] = {1, i};
  Run(et_->ComponentsBegin());
}

template <typename OutEdgeList, typename VertexList, typename... Args>
inline EliminationTree::Result DynamicCPUImprov::GetTDDecomp(
    std::size_t code,
    boost::adjacency_list<OutEdgeList,
                          VertexList,
                          boost::undirectedS,
                          Args...> const& induced_graph) {
  BoostGraph g;
  boost::copy_graph(induced_graph, g);
  return GetTDDecompImpl(code, g);
}
}  // namespace td
