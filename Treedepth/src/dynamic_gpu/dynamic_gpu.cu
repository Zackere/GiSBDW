// Copyright 2020 GISBDW. All rights reserved.
#include "dynamic_gpu.hpp"
// clang-format on

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/generate.h>
#include <algorithm>
#include <boost/graph/iteration_macros.hpp>

namespace td {
namespace {
__device__ int GetComponents(int8_t* component_belong_info,
                             int n,
                             int const* const offsets,
                             int const* const out_edges) {
  int ncomponent = 0;
  for (int v = 0; v < n; ++v) {
    if (component_belong_info[v] == -2) {
      component_belong_info[v] = ncomponent;
      bool new_vertex_found = true;
      while (new_vertex_found) {
        new_vertex_found = false;
        for (int w = 0; w < n; ++w)
          if (component_belong_info[w] == ncomponent) {
            for (int i = offsets[w]; i < offsets[w + 1]; ++i) {
              if (component_belong_info[out_edges[i]] == -2) {
                component_belong_info[out_edges[i]] = ncomponent;
                new_vertex_found = true;
              }
            }
          }
      }
      ncomponent++;
    }
  }
  return ncomponent;
}

__global__ void DynamicStepKernel(int2* const history,
                                  std::size_t const thread_offset,
                                  std::size_t const step_number,
                                  std::size_t const* const nk,
                                  int const n,
                                  int const* const offsets,
                                  int const* const out_edges) {
  extern __shared__ int8_t buf_shared[];
  int8_t* component_belong_info = &buf_shared[n * threadIdx.x];
  int set_code = thread_offset + blockDim.x * blockIdx.x + threadIdx.x;
  for (int i = 0; i < n; ++i)
    component_belong_info[i] = -1;
  set_encoder::Decode(n, step_number, set_code, component_belong_info);
  for (int i = 0; i < n; ++i) {
    while (component_belong_info[i] != i && component_belong_info[i] != -1) {
      auto tmp = component_belong_info[component_belong_info[i]];
      component_belong_info[component_belong_info[i]] =
          component_belong_info[i];
      component_belong_info[i] = tmp;
    }
  }
  for (int i = 0; i < n; ++i)
    component_belong_info[i] = component_belong_info[i] == -1 ? -1 : -2;
  int ncomponent = GetComponents(component_belong_info, n, offsets, out_edges);
  if (ncomponent == 1) {
    std::size_t graph_code = 0;
    std::size_t graph_verts = 0;
    for (int k = 0; k < n; ++k)
      if (component_belong_info[k] == 0)
        graph_code += set_encoder::NChooseK(k, ++graph_verts);
    int2 tdinfo = {n + 1, -1};
    for (int v = 0; v < n; ++v) {
      if (component_belong_info[v] < 0)
        continue;
      std::size_t code = 0;
      std::size_t element_index = 0;
      for (int k = 0; k < n; ++k)
        if (component_belong_info[k] == 0 && k != v)
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
        component_vertices += component_belong_info[j] == comp_num;

      for (int j = 0; j < component_vertices; ++j) {
        std::size_t code = 0;
        std::size_t x = 0;
        for (int k = 0; k < n; ++k)
          if (component_belong_info[k] == comp_num)
            code += set_encoder::NChooseK(k, ++x);
        if (history[nk[component_vertices] + code].x > tdinfo.x)
          tdinfo = history[nk[component_vertices] + code];
      }
    }
    std::size_t code = 0;
    std::size_t i = 0;
    for (int k = 0; k < n; ++k)
      if (component_belong_info[k] >= 0)
        code += set_encoder::NChooseK(k, ++i);
    history[nk[step_number] + code] = tdinfo;
  }
}
std::vector<cudaStream_t> DynamicStep(thrust::device_ptr<int2> history,
                                      std::size_t const shared_mem_per_thread,
                                      std::size_t const total_threads_to_launch,
                                      std::size_t const step_number,
                                      thrust::device_ptr<std::size_t> nk,
                                      int const n,
                                      thrust::device_ptr<int> offsets,
                                      thrust::device_ptr<int> out_edges) {
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
    DynamicStepKernel<<<kBlocks, kThreads, shared_mem_per_thread * kThreads,
                        streams.back()>>>(
        thrust::raw_pointer_cast(history), threads_launched, step_number,
        thrust::raw_pointer_cast(nk), n, thrust::raw_pointer_cast(offsets),
        thrust::raw_pointer_cast(out_edges));
  }
  if (excess >= kThreads) {
    streams.emplace_back();
    cudaStreamCreate(&streams.back());
    DynamicStepKernel<<<excess / kThreads, kThreads,
                        shared_mem_per_thread * kThreads, streams.back()>>>(
        thrust::raw_pointer_cast(history), threads_launched, step_number,
        thrust::raw_pointer_cast(nk), n, thrust::raw_pointer_cast(offsets),
        thrust::raw_pointer_cast(out_edges));
  }
  threads_launched += (excess / kThreads) * kThreads;
  excess %= kThreads;
  if (excess > 0) {
    streams.emplace_back();
    cudaStreamCreate(&streams.back());
    DynamicStepKernel<<<1, excess, shared_mem_per_thread * excess,
                        streams.back()>>>(
        thrust::raw_pointer_cast(history), threads_launched, step_number,
        thrust::raw_pointer_cast(nk), n, thrust::raw_pointer_cast(offsets),
        thrust::raw_pointer_cast(out_edges));
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
  thrust::device_vector<int> offsets(n + 1, 0);
  thrust::device_vector<int> out_edge(2 * boost::num_edges(g), 0);
  int offset = 0;
  for (int v = 0; v < n; ++v) {
    offsets[v] = offset;
    BGL_FORALL_ADJ_T(v, neigh, g, DynamicGPU::BoostGraph) {
      out_edge[offset++] = neigh;
    }
  }
  offsets[n] = 2 * boost::num_edges(g);
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
    SyncStreams(DynamicStep(history.data(), n, nk[i + 1] - nk[i], i, nk.data(),
                            n, offsets.data(), out_edge.data()));
  }
  history_.resize(history.size());
  thrust::copy(history.begin(), history.end(), history_.begin());
  nk_.resize(nk.size());
  thrust::copy(nk.begin(), nk.end(), nk_.begin());
}
}  // namespace td
