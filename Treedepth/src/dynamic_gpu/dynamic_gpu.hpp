// Copyright 2020 GISBDW. All rights reserved.
#pragma once
#ifdef CUDA_ENABLED
#include <memory>
#include <vector>

#include "boost/graph/adjacency_list.hpp"
#include "cuda_runtime.h"

#define H __host__
#define D __device__
#define HD H D

namespace td {
class DynamicGPU {
 public:
  // Refer to nvGraph (nvgraphCSRTopology32I_t) for details
  struct Graph {
    HD Graph(int verts, int edgs);
    HD Graph(Graph const& other);
    HD Graph& operator=(Graph const& other);
    HD Graph(Graph&&) = default;
    HD Graph& operator=(Graph&&) = default;
    HD ~Graph();

    int nvertices, nedges;
    int* source_offsets;
    int* destination;
  };

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

  std::size_t GetMaxIterations(std::size_t nvertices) const;

 private:
  struct HistoryEntry {
    std::unique_ptr<int8_t[]> uf;
    int8_t uf_size;
    int8_t vertex_added;

    HistoryEntry(int8_t size);
    HistoryEntry(HistoryEntry const& other);
    HistoryEntry& operator=(HistoryEntry const& other);
    HistoryEntry(HistoryEntry&&) = default;
    HistoryEntry& operator=(HistoryEntry&&) = default;
    ~HistoryEntry() = default;
  };
  void Run(Graph g, int k);

  std::vector<std::vector<HistoryEntry>> history_;
};

template <typename OutEdgeList, typename VertexList, typename... Args>
inline void DynamicGPU::operator()(boost::adjacency_list<OutEdgeList,
                                                         VertexList,
                                                         boost::undirectedS,
                                                         Args...> const& g) {
  return operator()(g, GetMaxIterations(boost::num_vertices(g)));
}
template <typename OutEdgeList, typename VertexList, typename... Args>
inline void DynamicGPU::operator()(boost::adjacency_list<OutEdgeList,
                                                         VertexList,
                                                         boost::undirectedS,
                                                         Args...> const& g,
                                   int k) {
  Graph copy(boost::num_vertices(g), boost::num_edges(g));
  int offset = 0;
  typename boost::graph_traits<
      std::remove_reference_t<decltype(g)>>::out_edge_iterator ei,
      ei_end;
  for (int i = 0; i < copy.nvertices; ++i) {
    copy.source_offsets[i] = offset;
    for (boost::tie(ei, ei_end) = boost::out_edges(i, g); ei != ei_end; ++ei)
      copy.destination[offset++] = boost::target(*ei, g);
  }
  copy.source_offsets[copy.nvertices] = offset;
  Run(copy, k);
}
}  // namespace td
#undef HD
#undef D
#undef H
#endif
