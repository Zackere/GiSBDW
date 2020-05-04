// Copyright 2020 GISBDW. All rights reserved.
#include "dynamic_gpu.hpp"
// clang-format on
#ifdef CUDA_ENABLED

#include <algorithm>

#include "cooperative_groups.h"

namespace td {
namespace {
__host__ __device__ std::size_t NChooseK(int n, int k) {
  if (k < 0 || k > n)
    return 0;
  if (k > n - k)
    k = n - k;
  std::size_t ret = 1;
  for (int i = 1; i <= k; ++i, --n) {
    ret *= n;
    ret /= i;
  }
  return ret;
}
__host__ __device__ std::size_t Encode(int8_t* sorted_set, int n, int k) {
  std::size_t ret = 0;
  for (int i = 0; i < k; ++i)
    ret += NChooseK(sorted_set[i], i + 1);
  return ret;
}
__host__ __device__ std::size_t Encode(int8_t* sorted_set,
                                       int n,
                                       int k,
                                       int exclude) {
  std::size_t ret = 0;
  int i = -1;
  while (++i < exclude)
    ret += NChooseK(sorted_set[i], i + 1);
  while (++i < k)
    ret += NChooseK(sorted_set[i], i + 1);
  return ret;
}
__host__ __device__ int8_t* Decode(std::size_t code, int n, int k) {
  int8_t* ret = new int8_t[k];
  int i = -1;
  while (k > 0) {
    auto nk = NChooseK(--n, k);
    if (code >= nk) {
      ret[++i] = n;
      code -= nk;
      --k;
    }
  }
  return ret;
}

__global__ void TDDynamicStep(int8_t* prev,
                              int8_t* next,
                              std::size_t step_number) {
  namespace cg = cooperative_groups;
  auto block = cg::this_thread_block();
  printf("%d\n", block.thread_rank());
}
}  // namespace

std::size_t DynamicGPU::GetMaxIterations(std::size_t nvertices) const {
  return nvertices;
}

void DynamicGPU::Run(Graph g, int k) {
  TDDynamicStep<<<4, 8>>>(nullptr, nullptr, 0);
}

DynamicGPU::Graph::Graph(int verts, int edgs) : nvertices(verts), nedges(edgs) {
  source_offsets = new int[nvertices + 1];
  destination = new int[nedges];
}

DynamicGPU::Graph::Graph(Graph const& other)
    : nvertices(other.nvertices),
      nedges(other.nedges),
      source_offsets(new int[nvertices + 1]),
      destination(new int[nedges]) {
  std::copy(other.source_offsets, other.source_offsets + other.nvertices + 1,
            source_offsets);
  std::copy(other.destination, other.destination + other.nedges, destination);
}

DynamicGPU::Graph& DynamicGPU::Graph::operator=(Graph const& other) {
  if (this != &other) {
    delete[] source_offsets;
    delete[] destination;
    nvertices = other.nvertices;
    nedges = other.nedges;
    source_offsets = new int[nvertices + 1];
    destination = new int[nedges];
    std::copy(other.source_offsets, other.source_offsets + other.nvertices + 1,
              source_offsets);
    std::copy(other.destination, other.destination + other.nedges, destination);
  }
  return *this;
}

DynamicGPU::Graph::~Graph() {
  delete[] source_offsets;
  delete[] destination;
}
}  // namespace td
#endif
