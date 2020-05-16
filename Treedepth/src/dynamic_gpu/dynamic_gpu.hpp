// Copyright 2020 GISBDW. All rights reserved.
#pragma once
#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/copy.hpp>
#include <list>
#include <mutex>
#include <set>
#include <vector>

#include "../set_encoder/set_encoder.hpp"

namespace td {
class DynamicGPU {
 public:
  using BoostGraph =
      boost::adjacency_list<boost::mapS, boost::vecS, boost::undirectedS>;
  using VertexType = int8_t;
  using OffsetType = uint16_t;

  template <typename OutEdgeList, typename VertexList, typename... Args>
  void operator()(boost::adjacency_list<OutEdgeList,
                                        VertexList,
                                        boost::undirectedS,
                                        Args...> const& g);
  template <typename OutEdgeList, typename VertexList, typename... Args>
  void operator()(boost::adjacency_list<OutEdgeList,
                                        VertexList,
                                        boost::undirectedS,
                                        Args...> const& g,
                  std::size_t k);

  std::size_t GetMaxIterations(std::size_t nvertices,
                               std::size_t nedges,
                               int device) const;
  std::size_t GetIterationsPerformed() const;

  template <typename VertexType>
  std::list<VertexType> GetElimination(std::size_t nverts,
                                       std::size_t subset_size,
                                       std::size_t subset_code);
  unsigned GetTreedepth(std::size_t nverts,
                        std::size_t subset_size,
                        std::size_t subset_code);

 private:
  std::size_t SetPlaceholderSize(std::size_t nverts) const;
  std::size_t SharedMemoryPerThread(std::size_t nverts,
                                    std::size_t step_num) const;
  std::size_t GlobalMemoryForStep(std::size_t nverts,
                                  std::size_t nedges,
                                  std::size_t step_num) const;
  void Run(BoostGraph const& in, std::size_t k);

  std::vector<std::vector<VertexType>> history_;
  std::vector<std::mutex> history_mtx_;
};

template <typename OutEdgeList, typename VertexList, typename... Args>
inline void DynamicGPU::operator()(boost::adjacency_list<OutEdgeList,
                                                         VertexList,
                                                         boost::undirectedS,
                                                         Args...> const& g) {
  return operator()(
      g, GetMaxIterations(boost::num_vertices(g), boost::num_edges(g), 0));
}

template <typename OutEdgeList, typename VertexList, typename... Args>
inline void DynamicGPU::operator()(boost::adjacency_list<OutEdgeList,
                                                         VertexList,
                                                         boost::undirectedS,
                                                         Args...> const& g,
                                   std::size_t k) {
  BoostGraph copy;
  boost::copy_graph(g, copy);
  Run(copy, k);
}

template <typename VertexType>
inline std::list<VertexType> DynamicGPU::GetElimination(
    std::size_t nverts,
    std::size_t subset_size,
    std::size_t subset_code) {
  if (subset_size > history_.size())
    return {};
  std::list<VertexType> ret;
  std::unique_lock<std::mutex>{history_mtx_[subset_size]};
  auto vertices = set_encoder::Decode<std::set<VertexType>>(nverts, subset_size,
                                                            subset_code);
  for (std::size_t i = 0; i < subset_size; ++i) {
    auto code = set_encoder::Encode(vertices);
    ret.push_back(history_[vertices.size()][code]);
    vertices.erase(history_[vertices.size()][code]);
  }
  return ret;
}
}  // namespace td
#endif
