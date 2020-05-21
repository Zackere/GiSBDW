// Copyright 2020 GISBDW. All rights reserved.
#include "dynamic_gpu.hpp"
// clang-format on
#ifdef CUDA_ENABLED

#include <thrust/device_vector.h>

#include <algorithm>
#include <utility>

#include "src/union_find/ext_array_union_find.hpp"

namespace td {
namespace {
// Refer to nvGraph (nvgraphCSRTopology32I_t) for details
template <typename VertexType, typename OffsetType>
struct Graph {
  std::size_t nvertices;
  thrust::device_vector<OffsetType> source_offsets;
  thrust::device_vector<VertexType> destination;

  explicit Graph(DynamicGPU::BoostGraph const& g)
      : nvertices(boost::num_vertices(g)),
        source_offsets(nvertices + 1),
        destination(2 * boost::num_edges(g)) {
    OffsetType offset = 0;
    typename boost::graph_traits<
        std::remove_reference_t<decltype(g)>>::out_edge_iterator ei,
        ei_end;
    for (VertexType i = 0; i < nvertices; ++i) {
      source_offsets[i] = offset;
      for (boost::tie(ei, ei_end) = boost::out_edges(i, g); ei != ei_end; ++ei)
        destination[offset++] = boost::target(*ei, g);
    }
    source_offsets[nvertices] = offset;
  }
};

template <typename VertexType, typename OffsetType>
__device__ bool BinarySearch(VertexType const* sorted_set,
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
  set_encoder::Decode(nvertices, step_number,
                      thread_offset + blockDim.x * blockIdx.x + threadIdx.x,
                      &mem[threadIdx.x * step_number]);
  auto const* const my_set = &mem[threadIdx.x * step_number];
  auto* const prev_uf =
      &mem[blockDim.x * step_number + threadIdx.x * (nvertices + 2)];
  VertexType current_best_td = step_number + 1;
  for (std::size_t i = 0; i < step_number; ++i) {
    auto code = set_encoder::Encode(my_set, step_number, i);
    for (std::size_t j = 0; j < nvertices + 2; ++j)
      prev_uf[j] = prev[code + j * prev_size];
    prev_uf[nvertices] = my_set[i];
    for (OffsetType off = source_offsets[my_set[i]];
         off < source_offsets[my_set[i] + 1]; ++off)
      if (BinarySearch(my_set, step_number, destination[off]))
        ext_array_union_find::Union(
            prev_uf, my_set[i],
            ext_array_union_find::Find(prev_uf, destination[off]),
            nvertices + 1);
    if (prev_uf[nvertices + 1] < current_best_td) {
      current_best_td = prev_uf[nvertices + 1];
      for (std::size_t j = 0; j < nvertices + 2; ++j)
        next[thread_offset + blockDim.x * blockIdx.x + threadIdx.x +
             j * next_size] = prev_uf[j];
    }
  }
}

template <typename VertexType, typename OffsetType>
std::vector<cudaStream_t> DynamicStep(VertexType const* const prev,
                                      VertexType* const next,
                                      std::size_t const shared_mem_per_thread,
                                      std::size_t const step_number,
                                      std::size_t const prev_size,
                                      std::size_t const next_size,
                                      Graph<VertexType, OffsetType> const& g) {
  // Adjust those values according to gpu specs
  constexpr std::size_t kThreads = 128, kBlocks = 4096,
                        kGridSize = kThreads * kBlocks;
  std::size_t excess = next_size % kGridSize;
  std::size_t max = next_size - excess;
  std::size_t threads_launched = 0;
  std::vector<cudaStream_t> streams;
  streams.reserve(max / kGridSize + 2);

  for (threads_launched = 0; threads_launched < max;
       threads_launched += kGridSize) {
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
  if (subset_size >= history_.size())
    return 0;
  std::unique_lock<std::mutex>{history_mtx_[subset_size]};
  return history_[subset_size][subset_code + history_[subset_size].size() / 2];
}

std::size_t DynamicGPU::SetPlaceholderSize(std::size_t nverts) const {
  return nverts + 2;
}

std::size_t DynamicGPU::SharedMemoryPerThread(std::size_t nverts,
                                              std::size_t step_num) const {
  return (SetPlaceholderSize(nverts) + step_num) * sizeof(VertexType);
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
  thrust::device_vector<VertexType> d_prev, d_next;
  if (k > g.nvertices / 2 && g.nvertices % 2) {
    d_next.reserve(set_encoder::NChooseK(g.nvertices, g.nvertices / 2) *
                   SetPlaceholderSize(g.nvertices));
    d_prev.reserve(d_next.capacity());
  } else {
    auto ktmp = std::min(k, g.nvertices / 2);
    d_next.reserve(set_encoder::NChooseK(g.nvertices, ktmp - 1) *
                   SetPlaceholderSize(g.nvertices));
    d_prev.reserve(set_encoder::NChooseK(g.nvertices, ktmp) *
                   SetPlaceholderSize(g.nvertices));
    if (ktmp % 2)
      d_next.swap(d_prev);
  }
  d_prev.resize(SetPlaceholderSize(g.nvertices), -1);
  d_prev[g.nvertices + 1] = 1;

  history_mtx_ = std::vector<std::mutex>(k);
  for (auto& mtx : history_mtx_)
    mtx.lock();
  history_ = decltype(history_)();
  history_.resize(k);

  history_[0].resize(2);
  history_[0][0] = -1;
  history_[0][1] = 0;
  history_mtx_[0].unlock();

  for (std::size_t i = 1; i < history_.size(); ++i) {
    d_next.resize(set_encoder::NChooseK(g.nvertices, i) *
                  SetPlaceholderSize(g.nvertices));
    SyncStreams(DynamicStep(thrust::raw_pointer_cast(d_prev.data()),
                            thrust::raw_pointer_cast(d_next.data()),
                            SharedMemoryPerThread(g.nvertices, i), i,
                            d_prev.size() / SetPlaceholderSize(g.nvertices),
                            d_next.size() / SetPlaceholderSize(g.nvertices),
                            g));
    history_[i].resize(2 * (d_next.size() / SetPlaceholderSize(g.nvertices)));
    thrust::copy(
        std::begin(d_next) + (history_[i].size() / 2) * g.nvertices,
        std::begin(d_next) + (history_[i].size() / 2) * (g.nvertices + 2),
        std::begin(history_[i]));
    history_mtx_[i].unlock();
    d_next.swap(d_prev);
  }
}
}  // namespace td
#endif
