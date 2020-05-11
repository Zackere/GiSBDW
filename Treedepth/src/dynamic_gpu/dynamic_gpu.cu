// Copyright 2020 GISBDW. All rights reserved.
#include "dynamic_gpu.hpp"
// clang-format on
#ifdef CUDA_ENABLED

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <utility>

#include "../set_encoder/set_encoder.hpp"
#include "../union_find/ext_array_union_find.hpp"
#include "cooperative_groups.h"

namespace td {
namespace {
__device__ bool BinarySearch(int8_t* sorted_set, int8_t size, int8_t val) {
  int8_t low = 0;
  int8_t high = size - 1;
  while (low <= high) {
    int8_t mid = (low + high) / 2;
    if (sorted_set[mid] > val)
      high = mid - 1;
    else if (sorted_set[mid] < val)
      low = mid + 1;
    else
      return true;
  }
  return false;
}

__global__ void DynamicStepKernel(int8_t const* const prev,
                                  int8_t* const next,
                                  std::size_t const thread_offset,
                                  std::size_t const prev_size,
                                  std::size_t const next_size,
                                  int const step_number,
                                  int const nvertices,
                                  int const* const source_offsets,
                                  int const* const destination) {
  extern __shared__ int8_t mem[];
  int8_t* my_uf = &mem[threadIdx.x * (nvertices + 2)];
  my_uf[nvertices + 1] = nvertices + 1;
  int8_t* my_set =
      &mem[blockDim.x * (nvertices + 2) + threadIdx.x * (step_number + 1)];
  set_encoder::Decode(thread_offset + blockIdx.x * blockDim.x + threadIdx.x,
                      nvertices, step_number + 1, my_set);
  int8_t* prev_uf = &mem[blockDim.x * (nvertices + 2 + step_number + 1) +
                         threadIdx.x * (nvertices + 2)];
  for (int i = 0; i <= step_number; ++i) {
    auto code = set_encoder::Encode(my_set, step_number + 1, i);
    for (int j = 0; j < nvertices + 2; ++j)
      prev_uf[j] = prev[code + j * prev_size];
    for (int off = source_offsets[my_set[i]];
         off < source_offsets[my_set[i] + 1]; ++off)
      if (BinarySearch(my_set, step_number + 1, destination[off]))
        ext_array_union_find::Union<int8_t>(
            prev_uf, my_set[i],
            ext_array_union_find::Find<int8_t>(prev_uf, destination[off]),
            nvertices + 1);
    if (prev_uf[nvertices + 1] < my_uf[nvertices + 1]) {
      prev_uf[nvertices] = my_set[i];
      auto* tmp = my_uf;
      my_uf = prev_uf;
      prev_uf = tmp;
    }
  }
  __syncthreads();
  for (int i = 0; i < nvertices + 2; ++i)
    next[thread_offset + blockIdx.x * blockDim.x + threadIdx.x +
         i * next_size] = my_uf[i];
}
std::vector<cudaStream_t> DynamicStep(int8_t* prev,
                                      int8_t* next,
                                      std::size_t shared_mem_per_thread,
                                      int step_number,
                                      std::size_t prev_size,
                                      std::size_t next_size,
                                      DynamicGPU::Graph const& g) {
  // Adjust those values according to gpu specs
  constexpr int kThreads = 64, kBlocks = 4096, kGridSize = kThreads * kBlocks;
  int excess = next_size % kGridSize;
  int max = next_size - excess;
  int threads_launched = 0;
  std::vector<cudaStream_t> streams;
  streams.reserve(max / kGridSize + 2);

  for (; threads_launched < max; threads_launched += kGridSize) {
    streams.emplace_back();
    cudaStreamCreate(&streams.back());
    DynamicStepKernel<<<kBlocks, kThreads, shared_mem_per_thread * kThreads,
                        streams.back()>>>(
        prev, next, threads_launched, prev_size, next_size, step_number,
        g.nvertices, thrust::raw_pointer_cast(g.source_offsets.data()),
        thrust::raw_pointer_cast(g.destination.data()));
  }
  if (excess >= kThreads) {
    streams.emplace_back();
    cudaStreamCreate(&streams.back());
    DynamicStepKernel<<<excess / kThreads, kThreads,
                        shared_mem_per_thread * kThreads, streams.back()>>>(
        prev, next, threads_launched, prev_size, next_size, step_number,
        g.nvertices, thrust::raw_pointer_cast(g.source_offsets.data()),
        thrust::raw_pointer_cast(g.destination.data()));
  }
  threads_launched += (excess / kThreads) * kThreads;
  excess %= kThreads;
  if (excess > 0) {
    streams.emplace_back();
    cudaStreamCreate(&streams.back());
    DynamicStepKernel<<<1, excess, shared_mem_per_thread * excess,
                        streams.back()>>>(
        prev, next, threads_launched, prev_size, next_size, step_number,
        g.nvertices, thrust::raw_pointer_cast(g.source_offsets.data()),
        thrust::raw_pointer_cast(g.destination.data()));
  }
  return streams;
}

void SyncStreams(std::vector<cudaStream_t> const& streams) {
  for (auto stream : streams) {
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
  }
}
}  // namespace

std::size_t DynamicGPU::GetMaxIterations(std::size_t nvertices,
                                         int device) const {
  cudaSetDevice(device);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);
  std::size_t shared_mem = deviceProp.sharedMemPerBlock;
  std::size_t global_mem = deviceProp.totalGlobalMem;
  std::size_t step = 0;
  while (true) {
    if (step == nvertices + 1)
      break;
    if (shared_mem < SharedMemoryPerThread(nvertices, step))
      break;
    if (global_mem < GlobalMemoryForStep(nvertices, step))
      break;
    ++step;
  }
  return step;
}

std::vector<int8_t> DynamicGPU::GetElimination(std::size_t nverts,
                                               std::size_t subset_size,
                                               std::size_t subset_code) {
  if (subset_size > history_.size())
    return {};
  std::vector<int8_t> ret(subset_size);
  std::unique_lock<std::mutex>{history_mtx_[subset_size]};
  auto vertices = set_encoder::Decode<int8_t>(nverts, subset_size, subset_code);
  for (int i = 0; i < ret.size(); ++i) {
    auto code = set_encoder::Encode(vertices);
    ret[i] = history_[vertices.size()][code];
    vertices.erase(history_[vertices.size()][code]);
  }
  return ret;
}

std::size_t DynamicGPU::SetPlaceholderSize(std::size_t nverts) const {
  return (nverts + 2) * sizeof(int8_t);
}

std::size_t DynamicGPU::SharedMemoryPerThread(std::size_t nverts,
                                              std::size_t step_num) const {
  return (2 * SetPlaceholderSize(nverts) + step_num + 1) * sizeof(int8_t);
}

std::size_t DynamicGPU::GlobalMemoryForStep(std::size_t nverts,
                                            std::size_t step_num) const {
  return (td::set_encoder::NChooseK(nverts, step_num) +
          td::set_encoder::NChooseK(nverts, step_num + 1)) *
         SetPlaceholderSize(nverts);
}

DynamicGPU::Graph DynamicGPU::Convert(BoostGraph const& g) {
  Graph copy{boost::num_vertices(g), 2 * boost::num_edges(g)};
  copy.source_offsets.resize(copy.nvertices + 1);
  copy.destination.resize(copy.nedges);
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
  return copy;
}

void DynamicGPU::Run(Graph const& g, int k) {
  thrust::device_vector<int8_t> d_prev(
      set_encoder::NChooseK(g.nvertices, 0) * SetPlaceholderSize(g.nvertices),
      -1);
  thrust::device_vector<int8_t> d_next;
  d_prev[g.nvertices + 1] = 1;
  history_mtx_ = std::vector<std::mutex>(k);
  for (auto& mtx : history_mtx_)
    mtx.lock();
  history_.clear();
  history_.resize(k);

  for (int i = 0; i < history_.size(); ++i) {
    d_next.clear();
    d_next.shrink_to_fit();
    d_next.resize(set_encoder::NChooseK(g.nvertices, i + 1) *
                  SetPlaceholderSize(g.nvertices));
    SyncStreams(DynamicStep(thrust::raw_pointer_cast(d_prev.data()),
                            thrust::raw_pointer_cast(d_next.data()),
                            SharedMemoryPerThread(g.nvertices, i), i,
                            d_prev.size() / SetPlaceholderSize(g.nvertices),
                            d_next.size() / SetPlaceholderSize(g.nvertices),
                            g));
    history_[i].resize(2 * d_prev.size() / SetPlaceholderSize(g.nvertices));
    thrust::copy(
        std::begin(d_prev) + history_[i].size() * g.nvertices / 2,
        std::begin(d_prev) + history_[i].size() * (g.nvertices + 2) / 2,
        std::begin(history_[i]));
    history_mtx_[i].unlock();
    d_prev = std::move(d_next);
  }
}
}  // namespace td
#endif
