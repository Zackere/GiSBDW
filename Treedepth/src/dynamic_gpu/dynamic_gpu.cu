// Copyright 2020 GISBDW. All rights reserved.
#include "dynamic_gpu.hpp"
// clang-format on
#ifdef CUDA_ENABLED

#include <thrust/device_vector.h>

#include <utility>

#include "../union_find/ext_array_union_find.hpp"

namespace td {
namespace {
// Refer to nvGraph (nvgraphCSRTopology32I_t) for details
template <typename VertexType, typename OffsetType>
struct Graph {
  std::size_t nvertices;
  std::size_t nedges;
  thrust::device_vector<OffsetType> source_offsets;
  thrust::device_vector<VertexType> destination;

  explicit Graph(DynamicGPU::BoostGraph const& g)
      : nvertices(boost::num_vertices(g)),
        nedges(2 * boost::num_edges(g)),
        source_offsets(nvertices + 1),
        destination(nedges) {
    OffsetType offset = 0;
    typename boost::graph_traits<
        std::remove_reference_t<decltype(g)>>::out_edge_iterator ei,
        ei_end;
    for (std::size_t i = 0; i < nvertices; ++i) {
      source_offsets[i] = offset;
      for (boost::tie(ei, ei_end) = boost::out_edges(i, g); ei != ei_end; ++ei)
        destination[offset++] = boost::target(*ei, g);
    }
    source_offsets[nvertices] = offset;
  }
};

template <typename VertexType, typename OffsetType>
__device__ bool BinarySearch(VertexType* sorted_set,
                             OffsetType size,
                             VertexType val) {
  OffsetType low = 0;
  OffsetType high = size;
  while (low < high) {
    std::size_t mid = (low + high) / 2;
    if (sorted_set[mid] > val)
      high = mid;
    else if (sorted_set[mid] < val)
      low = mid + 1;
    else
      return true;
  }
  return false;
}
template <typename VertexType, typename OffsetType>
__global__ void DynamicStepKernel(VertexType const* const prev,
                                  VertexType* const next,
                                  std::size_t const thread_offset,
                                  std::size_t const prev_size,
                                  std::size_t const next_size,
                                  std::size_t const step_number,
                                  std::size_t const nvertices,
                                  OffsetType const* const source_offsets,
                                  VertexType const* const destination) {
  extern __shared__ VertexType mem[];
  auto* my_set = &mem[threadIdx.x * (step_number + 1)];
  set_encoder::Decode(nvertices, step_number + 1,
                      thread_offset + blockIdx.x * blockDim.x + threadIdx.x,
                      my_set);
  auto* prev_uf =
      &mem[blockDim.x * (step_number + 1) + threadIdx.x * (nvertices + 2)];
  next[thread_offset + blockIdx.x * blockDim.x + threadIdx.x +
       (nvertices + 1) * next_size] = nvertices + 1;
  for (std::size_t i = 0; i <= step_number; ++i) {
    auto code = set_encoder::Encode(my_set, step_number + 1, i);
    for (std::size_t j = 0; j < nvertices + 2; ++j)
      prev_uf[j] = prev[code + j * prev_size];
    for (OffsetType off = source_offsets[my_set[i]];
         off < source_offsets[my_set[i] + 1]; ++off)
      if (BinarySearch(my_set, step_number + 1, destination[off]))
        ext_array_union_find::Union<VertexType>(
            prev_uf, my_set[i],
            ext_array_union_find::Find(prev_uf, destination[off]),
            nvertices + 1);
    prev_uf[nvertices] = my_set[i];
    if (prev_uf[nvertices + 1] <
        next[thread_offset + blockIdx.x * blockDim.x + threadIdx.x +
             (nvertices + 1) * next_size])
      for (std::size_t i = 0; i < nvertices + 2; ++i)
        next[thread_offset + blockIdx.x * blockDim.x + threadIdx.x +
             i * next_size] = prev_uf[i];
  }
}

template <typename VertexType, typename OffsetType>
std::vector<cudaStream_t> DynamicStep(VertexType const* prev,
                                      VertexType* next,
                                      std::size_t shared_mem_per_thread,
                                      std::size_t step_number,
                                      std::size_t prev_size,
                                      std::size_t next_size,
                                      Graph<VertexType, OffsetType> const& g) {
  // Adjust those values according to gpu specs
  constexpr std::size_t kThreads = 128, kBlocks = 4096,
                        kGridSize = kThreads * kBlocks;
  std::size_t excess = next_size % kGridSize;
  std::size_t max = next_size - excess;
  std::size_t threads_launched = 0;
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
                                         std::size_t nedges,
                                         int device) const {
  cudaSetDevice(device);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);
  std::size_t shared_mem = deviceProp.sharedMemPerBlock;
  std::size_t global_mem;
  cudaMemGetInfo(&global_mem, nullptr);
  std::size_t step = 0;
  while (true) {
    if (shared_mem < SharedMemoryPerThread(nvertices, step))
      break;
    if (global_mem < GlobalMemoryForStep(nvertices, nedges, step))
      break;
    if (step == nvertices + 1)
      break;
    ++step;
  }
  return step;
}

std::size_t DynamicGPU::GetIterationsPerformed() const {
  return history_.size();
}

unsigned DynamicGPU::GetTreedepth(std::size_t nverts,
                                  std::size_t subset_size,
                                  std::size_t subset_code) {
  if (subset_size > history_.size())
    return 0;
  std::unique_lock<std::mutex>{history_mtx_[subset_size]};
  return history_[subset_size][subset_code + history_[subset_size].size() / 2];
}

std::size_t DynamicGPU::SetPlaceholderSize(std::size_t nverts) const {
  return nverts + 2;
}

std::size_t DynamicGPU::SharedMemoryPerThread(std::size_t nverts,
                                              std::size_t step_num) const {
  return (SetPlaceholderSize(nverts) + step_num + 1) * sizeof(VertexType);
}

std::size_t DynamicGPU::GlobalMemoryForStep(std::size_t nverts,
                                            std::size_t nedges,
                                            std::size_t step_num) const {
  return (td::set_encoder::NChooseK(nverts, step_num) +
          td::set_encoder::NChooseK(nverts, step_num + 1)) *
             SetPlaceholderSize(nverts) * sizeof(VertexType) +
         (nverts + 1) * sizeof(OffsetType) + 2 * nedges * sizeof(VertexType);
}

void DynamicGPU::Run(BoostGraph const& in, std::size_t k) {
  Graph<VertexType, OffsetType> g(in);
  thrust::device_vector<VertexType> d_prev(
      set_encoder::NChooseK(g.nvertices, 0) * SetPlaceholderSize(g.nvertices),
      -1);
  thrust::device_vector<VertexType> d_next;
  d_prev[g.nvertices + 1] = 1;
  history_mtx_ = std::vector<std::mutex>(k);
  for (auto& mtx : history_mtx_)
    mtx.lock();
  history_.clear();
  history_.resize(k);

  for (std::size_t i = 0; i < history_.size(); ++i) {
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
