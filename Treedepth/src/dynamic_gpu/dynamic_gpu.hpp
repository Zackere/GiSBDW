// Copyright 2020 GISBDW. All rights reserved.
#pragma once
#ifdef CUDA_ENABLED
#include <thrust/device_vector.h>

#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <vector>

#include "boost/graph/adjacency_list.hpp"
#include "boost/graph/copy.hpp"
#include "cuda_runtime.h"

namespace td {
class DynamicGPU {
 public:
  // Refer to nvGraph (nvgraphCSRTopology32I_t) for details
  struct Graph {
    int nvertices;
    int nedges;
    thrust::device_vector<int> source_offsets;
    thrust::device_vector<int> destination;
  };
  using BoostGraph =
      boost::adjacency_list<boost::mapS, boost::vecS, boost::undirectedS>;

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
                  int k);

  std::size_t GetMaxIterations(std::size_t nvertices, int device) const;
  std::vector<int8_t> GetElimination(std::set<int8_t> vertices,
                                     std::size_t nverts);

 private:
  std::size_t SetPlaceholderSize(std::size_t nverts) const;
  std::size_t SharedMemoryPerThread(std::size_t nverts,
                                    std::size_t step_num) const;
  std::size_t GlobalMemoryForStep(std::size_t nverts,
                                  std::size_t step_num) const;
  Graph Convert(BoostGraph const& g);
  void Run(Graph const& g, int k);

  std::vector<std::vector<int8_t>> history_;
  std::vector<std::mutex> history_mtx_;
};

template <typename OutEdgeList, typename VertexList, typename... Args>
inline void DynamicGPU::operator()(boost::adjacency_list<OutEdgeList,
                                                         VertexList,
                                                         boost::undirectedS,
                                                         Args...> const& g) {
  return operator()(g, GetMaxIterations(boost::num_vertices(g), 0));
}
template <typename OutEdgeList, typename VertexList, typename... Args>
inline void DynamicGPU::operator()(boost::adjacency_list<OutEdgeList,
                                                         VertexList,
                                                         boost::undirectedS,
                                                         Args...> const& g,
                                   int k) {
  BoostGraph copy;
  boost::copy_graph(g, copy);
  Run(Convert(copy), k);
}
}  // namespace td
#endif
