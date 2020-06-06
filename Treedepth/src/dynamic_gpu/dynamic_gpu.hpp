// Copyright 2020 GISBDW. All rights reserved.
#pragma once

#include <cuda_runtime.h>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/copy.hpp>

#include <vector>

#include "src/set_encoder/set_encoder.hpp"

namespace td {
class DynamicGPU {
 public:
  using BoostGraph =
      boost::adjacency_list<boost::mapS, boost::vecS, boost::undirectedS>;
  template <typename OutEdgeList, typename VertexList, typename... Args>
  void operator()(boost::adjacency_list<OutEdgeList,
                                        VertexList,
                                        boost::undirectedS,
                                        Args...> const& g);

  template <typename VertexType>
  inline std::list<VertexType> GetElimination(std::size_t nverts,
                                              std::size_t subset_size,
                                              std::size_t subset_code);

 private:
  void Run(BoostGraph const& g);
  int nverts_;
  std::vector<int2> history_;
  std::vector<std::size_t> nk_;
};

template <typename OutEdgeList, typename VertexList, typename... Args>
inline void DynamicGPU::operator()(boost::adjacency_list<OutEdgeList,
                                                         VertexList,
                                                         boost::undirectedS,
                                                         Args...> const& g) {
  BoostGraph copy;
  boost::copy_graph(g, copy);
  nverts_ = boost::num_vertices(g);
  Run(copy);
}

template <typename VertexType>
inline std::list<VertexType> DynamicGPU::GetElimination(
    std::size_t nverts,
    std::size_t subset_size,
    std::size_t subset_code) {
  if (subset_size > history_.size())
    return {};
  std::list<VertexType> ret;
  auto vertices = set_encoder::Decode<std::set<VertexType>>(nverts, subset_size,
                                                            subset_code);
  for (std::size_t i = 0; i < subset_size; ++i) {
    auto code = set_encoder::Encode(vertices);
    ret.push_back(history_[nk_[vertices.size()] + code].y);
    vertices.erase(history_[nk_[vertices.size()] + code].y);
  }
  return ret;
}
}  // namespace td