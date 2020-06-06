// Copyright 2020 GISBDW. All rights reserved.
#include "dynamic_gpu.hpp"
// clang-format on

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/generate.h>
#include <boost/graph/iteration_macros.hpp>

namespace td {
namespace {
__global__ void DynamicStepKernel(int2* const history,
                                  std::size_t const thread_offset,
                                  std::size_t const step_number,
                                  std::size_t const n,
                                  std::size_t const* const nk,
                                  bool const* const g) {
  extern __shared__ int8_t mem[];
  int8_t* graph = &mem[0];
  int8_t* my_buf = &mem[n + n * threadIdx.x];
  for (int i = threadIdx.x; i < n * n; i += blockDim.x)
    graph[i] = g[i];
  __syncthreads();
  int set_code = thread_offset + blockDim.x * blockIdx.x + threadIdx.x;
  for (int i = 0; i < n; ++i)
    my_buf[i] = -1;
  set_encoder::Decode(n, step_number, set_code, my_buf);
  for (int i = 0; i < n; ++i) {
    while (my_buf[i] != i && my_buf[i] != -1) {
      auto tmp = my_buf[my_buf[i]];
      my_buf[my_buf[i]] = my_buf[i];
      my_buf[i] = tmp;
    }
  }
  for (int i = 0; i < n; ++i)
    my_buf[i] = my_buf[i] == -1 ? -1 : -2;
  int ncomponent = 0;
  for (int v = 0; v < n; ++v) {
    if (my_buf[v] == -2) {
      my_buf[v] = ncomponent;
      bool new_vertex_found = true;
      while (new_vertex_found) {
        new_vertex_found = false;
        for (int w = 0; w < n; ++w)
          if (my_buf[w] == ncomponent) {
            for (int i = 0; i < n; ++i) {
              if (g[w * n + i] && my_buf[i] == -2) {
                my_buf[i] = ncomponent;
                new_vertex_found = true;
              }
            }
          }
      }
      ncomponent++;
    }
  }
  if (ncomponent == 1) {
    std::size_t graph_code = 0;
    std::size_t graph_verts = 0;
    for (int k = 0; k < n; ++k)
      if (my_buf[k] == 0)
        graph_code += set_encoder::NChooseK(k, ++graph_verts);
    int2 tdinfo = {n + 1, -1};
    for (int v = 0; v < n; ++v) {
      if (my_buf[v] < 0)
        continue;
      std::size_t code = 0;
      std::size_t element_index = 0;
      for (int k = 0; k < n; ++k)
        if (my_buf[k] == 0 && k != v)
          code += set_encoder::NChooseK(k, ++element_index);
      if (history[nk[graph_verts - 1] + code].x + 1 < tdinfo.x) {
        tdinfo.x = history[nk[graph_verts - 1] + code].x + 1;
        tdinfo.y = v;
      }
    }
    history[nk[graph_verts] + graph_code] = tdinfo;
  } else {
    int2 tdinfo = {-1, -1};
    for (int comp_num = 0; comp_num < ncomponent; ++comp_num) {
      int component_vertices = 0;
      for (int j = 0; j < n; ++j)
        component_vertices += my_buf[j] == comp_num;

      for (int j = 0; j < component_vertices; ++j) {
        std::size_t code = 0;
        std::size_t x = 0;
        for (int k = 0; k < n; ++k)
          if (my_buf[k] == comp_num)
            code += set_encoder::NChooseK(k, ++x);
        if (history[nk[component_vertices] + code].x > tdinfo.x)
          tdinfo = history[nk[component_vertices] + code];
      }
    }
    std::size_t code = 0;
    std::size_t i = 0;
    for (int k = 0; k < n; ++k)
      if (my_buf[k] >= 0)
        code += set_encoder::NChooseK(k, ++i);
    history[nk[step_number] + code] = tdinfo;
  }
}
std::vector<cudaStream_t> DynamicStep(thrust::device_ptr<int2> history,
                                      std::size_t const shared_mem_per_thread,
                                      std::size_t const common_shared_mem,
                                      std::size_t const total_threads_to_launch,
                                      std::size_t const step_number,
                                      std::size_t const nvertices,
                                      thrust::device_ptr<std::size_t> nk,
                                      thrust::device_ptr<bool> g) {
  // Adjust those values according to gpu specs
  constexpr std::size_t kThreads = 128, kBlocks = 4096,
                        kGridSize = kThreads * kBlocks;
  std::size_t excess = total_threads_to_launch % kGridSize;
  std::size_t max = total_threads_to_launch - excess;
  std::size_t threads_launched = 0;
  std::vector<cudaStream_t> streams;
  streams.reserve(max / kGridSize + 2);

  for (threads_launched = 0; threads_launched < max;
       threads_launched += kGridSize) {
    streams.emplace_back();
    cudaStreamCreate(&streams.back());
    DynamicStepKernel<<<kBlocks, kThreads,
                        shared_mem_per_thread * kThreads + common_shared_mem,
                        streams.back()>>>(
        thrust::raw_pointer_cast(history), threads_launched, step_number,
        nvertices, thrust::raw_pointer_cast(nk), thrust::raw_pointer_cast(g));
  }
  if (excess >= kThreads) {
    streams.emplace_back();
    cudaStreamCreate(&streams.back());
    DynamicStepKernel<<<excess / kThreads, kThreads,
                        shared_mem_per_thread * kThreads + common_shared_mem,
                        streams.back()>>>(
        thrust::raw_pointer_cast(history), threads_launched, step_number,
        nvertices, thrust::raw_pointer_cast(nk), thrust::raw_pointer_cast(g));
  }
  threads_launched += (excess / kThreads) * kThreads;
  excess %= kThreads;
  if (excess > 0) {
    streams.emplace_back();
    cudaStreamCreate(&streams.back());
    DynamicStepKernel<<<1, excess,
                        shared_mem_per_thread * excess + common_shared_mem,
                        streams.back()>>>(
        thrust::raw_pointer_cast(history), threads_launched, step_number,
        nvertices, thrust::raw_pointer_cast(nk), thrust::raw_pointer_cast(g));
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
void DynamicGPU::Run(BoostGraph const& g) {
  int n = boost::num_vertices(g);
  thrust::device_vector<bool> adjacency_matrix(n * n, false);
  for (int v = 0; v < n; ++v) {
    BGL_FORALL_ADJ_T(v, neigh, g, DynamicGPU::BoostGraph) {
      adjacency_matrix[v * n + neigh] = true;
    }
  }
  thrust::device_vector<int2> history(1 << n);
  history[0] = {0, -1};
  for (int i = 0; i < n; ++i)
    history[1 + i] = {1, i};

  thrust::device_vector<std::size_t> nk(n + 2);
  nk[0] = 0;
  for (int i = 1; i <= n; ++i)
    nk[i] = nk[i - 1] + set_encoder::NChooseK(n, i - 1);
  nk[n + 1] = 1 << n;
  for (int i = 2; i <= n; ++i) {
    SyncStreams(DynamicStep(history.data(), n, n * n, nk[i + 1] - nk[i], i, n,
                            nk.data(), adjacency_matrix.data()));
  }
  history_.resize(history.size());
  thrust::copy(history.begin(), history.end(), history_.begin());
  nk_.resize(nk.size());
  thrust::copy(nk.begin(), nk.end(), nk_.begin());
}
}  // namespace td
