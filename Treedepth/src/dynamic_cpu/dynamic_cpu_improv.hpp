// Copyright 2020 GISBDW. All rights reserved.
#pragma once

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/copy.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "src/elimination_tree/elimination_tree.hpp"
#include "src/set_encoder/set_encoder.hpp"

namespace td {
class DynamicCPUImprov {
 public:
  using BoostGraph =
      boost::adjacency_list<boost::mapS, boost::vecS, boost::undirectedS>;
  using CodeType = boost::multiprecision::uint128_t;
  static constexpr auto kMaxVerts = 128;

  template <typename OutEdgeList, typename VertexList, typename... Args>
  void operator()(boost::adjacency_list<OutEdgeList,
                                        VertexList,
                                        boost::undirectedS,
                                        Args...> const& g);

  std::size_t GetIterationsPerformed() const;

  template <typename OutEdgeList, typename VertexList, typename... Args>
  EliminationTree::Result GetTDDecomp(
      CodeType code,
      boost::adjacency_list<OutEdgeList,
                            VertexList,
                            boost::undirectedS,
                            Args...> const& induced_graph) const;

  std::size_t GetTreedepth(std::size_t nverts,
                           std::size_t subset_size,
                           CodeType subset_code) const;

 private:
  std::size_t Run(EliminationTree::ComponentIterator component);
  EliminationTree::Result GetTDDecompImpl(CodeType code,
                                          BoostGraph const& g) const;

  std::unique_ptr<EliminationTree> et_;
  std::vector<std::unordered_map<CodeType, std::tuple<std::size_t, int>>>
      history_;
};
template <typename OutEdgeList, typename VertexList, typename... Args>
inline void DynamicCPUImprov::operator()(
    boost::adjacency_list<OutEdgeList,
                          VertexList,
                          boost::undirectedS,
                          Args...> const& g) {
  try {
    decltype(history_)().swap(history_);
    if (kMaxVerts < boost::num_vertices(g))
      return;
    et_.reset(new td::EliminationTree(g));
    history_.resize(boost::num_vertices(g) + 1);
    history_[0][0] = {0, -1};
    for (std::size_t i = 0; i < boost::num_vertices(g); ++i)
      history_[1][static_cast<CodeType>(1) << i] = {1, i};
    Run(et_->ComponentsBegin());
  } catch (std::bad_alloc const&) {
    std::cout << "Out of memory\n";
    return;
  }
}

template <typename OutEdgeList, typename VertexList, typename... Args>
inline EliminationTree::Result DynamicCPUImprov::GetTDDecomp(
    CodeType code,
    boost::adjacency_list<OutEdgeList,
                          VertexList,
                          boost::undirectedS,
                          Args...> const& induced_graph) const {
  BoostGraph g;
  boost::copy_graph(induced_graph, g);
  return GetTDDecompImpl(code, g);
}
}  // namespace td
