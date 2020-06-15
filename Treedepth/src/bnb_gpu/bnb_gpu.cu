// Copyright 2020 GISBDW. All rights reserved.
#include "bnb_gpu.hpp"
// clang-format on

#include <cuda_runtime_api.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <algorithm>
#include <boost/graph/iteration_macros.hpp>
#include <memory>

namespace td {
namespace {
auto constexpr kThreads = 64;
auto constexpr kBlocks = 2048;
auto constexpr kBruteForceLen = 6;

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

__device__ int EliminatePermutation(int8_t* component_belong_info,
                                    int8_t* component_depth_info,
                                    int8_t* my_perm,
                                    int perm_len,
                                    int n,
                                    int const* const offsets,
                                    int const* const out_edges) {
  int ncomponent;
  for (int i = 0; i < n; ++i) {
    component_belong_info[i] = -2;
    component_depth_info[i] = 0;
  }
  for (int i = 0; i < perm_len; ++i) {
    auto belongs_to = component_belong_info[my_perm[i]];
    for (int j = 0; j < n; ++j)
      if (component_belong_info[j] == belongs_to)
        ++component_depth_info[j];
    for (int j = 0; j < n; ++j)
      component_belong_info[j] = -2;
    for (int j = 0; j <= i; ++j)
      component_belong_info[my_perm[j]] = -1;
    ncomponent = GetComponents(component_belong_info, n, offsets, out_edges);
  }
  return ncomponent;
}

__device__ int LowerBound(int8_t* component_belong_info,
                          int8_t* component_depth_info,
                          int ncomponent,
                          int n,
                          int const* const offsets,
                          int const* const out_edges) {
  int lower_bound = 0;
  int current_bound;
  for (int i = 0; i < ncomponent; ++i) {
    int nverts = 0;
    int nedges = 0;
    int vert = -1;
    for (int v = 0; v < n; ++v) {
      if (component_belong_info[v] != i)
        continue;
      ++nverts;
      for (int j = offsets[v]; j < offsets[v + 1]; ++j)
        if (component_belong_info[out_edges[j]] == i)
          ++nedges;

      vert = v;
    }
    nedges >>= 1;
    current_bound =
        std::ceil(0.5 + nverts -
                  std::sqrt(0.25 + nverts * nverts - nverts - 2 * nedges)) +
        component_depth_info[vert];

    if (current_bound > lower_bound)
      lower_bound = current_bound;
  }

  return lower_bound;
}

__device__ void FinishPermutation(int8_t* const component_belong_info,
                                  int8_t* const component_depth_info,
                                  int8_t* const perm,
                                  int perm_len,
                                  int n,
                                  int const* const offsets,
                                  int const* const out_edges,
                                  int* best_td) {
  auto cur_perm_len = perm_len;
  int ncomponent =
      EliminatePermutation(component_belong_info, component_depth_info, perm,
                           cur_perm_len, n, offsets, out_edges);
  if (LowerBound(component_belong_info, component_depth_info, ncomponent, n,
                 offsets, out_edges) >= *best_td)
    return;
  auto* taken = new int8_t[n + 1];
  for (int i = 0; i <= n; ++i)
    taken[i] = false;
  for (int i = 0; i < perm_len; ++i)
    taken[perm[i]] = true;
  while (cur_perm_len >= perm_len) {
    if (ncomponent + cur_perm_len == n) {
      int max_td = 0;
      for (int i = 0; i < n; ++i)
        max_td = max(
            max_td, component_depth_info[i] + (component_belong_info[i] != -1));
      atomicMin(best_td, max_td);
      --cur_perm_len;
    }
    do {
      if (perm[cur_perm_len] != -1) {
        taken[perm[cur_perm_len]] = false;
        ncomponent =
            EliminatePermutation(component_belong_info, component_depth_info,
                                 perm, cur_perm_len, n, offsets, out_edges);
      }
      do {
        ++perm[cur_perm_len];
      } while (perm[cur_perm_len] < n && taken[perm[cur_perm_len]] &&
               component_belong_info[perm[cur_perm_len]] != 0);
      if (perm[cur_perm_len] >= n)
        break;
      ncomponent =
          EliminatePermutation(component_belong_info, component_depth_info,
                               perm, cur_perm_len + 1, n, offsets, out_edges);
      if (LowerBound(component_belong_info, component_depth_info, ncomponent, n,
                     offsets, out_edges) < *best_td)
        break;
    } while (true);
    if (perm[cur_perm_len] >= n) {
      perm[cur_perm_len] = -1;
      --cur_perm_len;
      continue;
    }
    taken[perm[cur_perm_len]] = true;
    ++cur_perm_len;
  }
  delete[] taken;
}

__global__ void GenerateKernel(int* const in,
                               int* const buf,
                               int* stack_head,
                               int const n,
                               int const* const offsets,
                               int const* const out_edges,
                               int* best_td) {
  extern __shared__ int8_t buf_shared[];
  int8_t* perm = &buf_shared[threadIdx.x * n];
  for (int i = 0; i < n; ++i)
    buf_shared[i * blockDim.x + threadIdx.x] = -1;
  __syncthreads();
  for (int i = threadIdx.x; i < blockDim.x * n; i += blockDim.x)
    buf_shared[i] = in[i + blockIdx.x * blockDim.x * n];
  __syncthreads();
  int perm_len = -1;
  while (perm_len < n && perm[++perm_len] != -1) {
  }
  int8_t* component_depth_info = &buf_shared[threadIdx.x * n + blockDim.x * n];
  int8_t* component_belong_info =
      &buf_shared[threadIdx.x * n + 2 * blockDim.x * n];
  for (int i = 0; i < n; ++i) {
    component_belong_info[i] = 0;
    component_depth_info[i] = 0;
  }
  int ncomponent = 0;
  for (int i = 0; i < perm_len; ++i) {
    ncomponent = 0;
    auto belongs_to = component_belong_info[perm[i]];
    for (int j = 0; j < n; ++j)
      if (component_belong_info[j] == belongs_to)
        ++component_depth_info[j];
    for (int j = 0; j < n; ++j)
      component_belong_info[j] = -2;
    for (int j = 0; j <= i; ++j)
      component_belong_info[perm[j]] = -1;
    ncomponent = GetComponents(component_belong_info, n, offsets, out_edges);
  }
  if (LowerBound(component_belong_info, component_depth_info, ncomponent, n,
                 offsets, out_edges) >= *best_td)
    return;
  if (ncomponent + perm_len == n) {
    int max_td = 0;
    for (int i = 0; i < n; ++i)
      max_td = max(max_td,
                   component_depth_info[i] + (component_belong_info[i] != -1));
    atomicMin(best_td, max_td);
    return;
  }
  // sprobuj wsadzic nowe
  if (__syncthreads_and(perm_len > n - kBruteForceLen))
    return FinishPermutation(component_belong_info, component_depth_info, perm,
                             perm_len, n, offsets, out_edges, best_td);
  for (int i = 0; i < n; ++i) {
    int j;
    for (j = 0; j < n && perm[j] != -1; ++j) {
      if (i == perm[j])
        break;
    }
    if (j < n && perm[j] == -1 && component_belong_info[i] == 0) {
      perm[j] = i;
      auto new_perm_index = atomicAdd(stack_head, 1);
      for (int k = 0; k < n; ++k) {
        buf[new_perm_index * n + k] = perm[k];
      }
      perm[j] = -1;
    }
  }
}

__global__ void BruteForceKernel(int* const buf,
                                 int stack_head,
                                 int const n,
                                 int const* const offsets,
                                 int const* const out_edges,
                                 int* best_td) {
  auto my_perm_index = stack_head - blockIdx.x * blockDim.x - 1;
  extern __shared__ int8_t buf_shared[];
  int8_t* perm = &buf_shared[threadIdx.x * n];
  int8_t* component_depth_info = &buf_shared[threadIdx.x * n + blockDim.x * n];
  int8_t* component_belong_info =
      &buf_shared[threadIdx.x * n + 2 * blockDim.x * n];
  for (int i = 0; i < n; ++i)
    buf_shared[i * blockDim.x + threadIdx.x] = -1;
  __syncthreads();
  for (int i = threadIdx.x; i < blockDim.x * n; i += blockDim.x) {
    buf_shared[i] = buf[my_perm_index * n + i];
    buf[my_perm_index * n + i] = -1;
  }
  __syncthreads();
  int perm_len = -1;
  while (perm_len < n && perm[++perm_len] != -1) {
  }
  return FinishPermutation(component_belong_info, component_depth_info, perm,
                           perm_len, n, offsets, out_edges, best_td);
}
}  // namespace
int BnBGPU::Run(BoostGraph const& g, std::size_t heur_td) {
  auto n = boost::num_vertices(g);
  std::size_t global_mem;
  cudaMemGetInfo(&global_mem, nullptr);
  global_mem *= 0.9;
  global_mem /= sizeof(int);
  thrust::device_vector<int> buf((global_mem / n) * n, -1);
  for (int i = 0; i < n; ++i)
    buf[i * n] = i;
  thrust::device_vector<int> temporary_buf(kThreads * kBlocks * n, -1);
  thrust::device_vector<int> stack_head(1, n);
  thrust::device_vector<int> best_td(1, heur_td);
  thrust::device_vector<int> offsets(n + 1, 0);
  thrust::device_vector<int> out_edge(2 * boost::num_edges(g), 0);
  int offset = 0;
  for (int v = 0; v < n; ++v) {
    offsets[v] = offset;
    BGL_FORALL_ADJ_T(v, neigh, g, BnBGPU::BoostGraph) {
      out_edge[offset++] = neigh;
    }
  }
  offsets[n] = 2 * boost::num_edges(g);
  int perms = 0;
  while (stack_head[0] > 0) {
    while (stack_head[0] > 0 &&
           kThreads * kBlocks * n < buf.size() / n - stack_head[0]) {
      int sh = stack_head[0];
      // std::cout << sh << std::endl;
      if (sh < kThreads) {
        thrust::copy(thrust::device, buf.begin(),
                     buf.begin() + stack_head[0] * n, temporary_buf.begin());
        thrust::fill(thrust::device, buf.begin(),
                     buf.begin() + stack_head[0] * n, -1);
        stack_head[0] = 0;
        GenerateKernel<<<1, sh, 3 * sh * n * sizeof(int8_t)>>>(
            thrust::raw_pointer_cast(temporary_buf.data()),
            thrust::raw_pointer_cast(buf.data()),
            thrust::raw_pointer_cast(stack_head.data()), n,
            thrust::raw_pointer_cast(offsets.data()),
            thrust::raw_pointer_cast(out_edge.data()),
            thrust::raw_pointer_cast(best_td.data()));
        perms += sh;
      } else if (sh < kBlocks * kThreads) {
        thrust::copy(thrust::device, buf.begin(),
                     buf.begin() + stack_head[0] * n, temporary_buf.begin());
        thrust::fill(thrust::device, buf.begin(),
                     buf.begin() + stack_head[0] * n, -1);
        stack_head[0] = 0;
        GenerateKernel<<<sh / kThreads, kThreads,
                         3 * kThreads * n * sizeof(int8_t)>>>(
            thrust::raw_pointer_cast(temporary_buf.data()),
            thrust::raw_pointer_cast(buf.data()),
            thrust::raw_pointer_cast(stack_head.data()), n,
            thrust::raw_pointer_cast(offsets.data()),
            thrust::raw_pointer_cast(out_edge.data()),
            thrust::raw_pointer_cast(best_td.data()));
        perms += (sh / kThreads) * kThreads;
      } else {
        thrust::copy(thrust::device,
                     buf.begin() + stack_head[0] * n - kThreads * kBlocks * n,
                     buf.begin() + stack_head[0] * n, temporary_buf.begin());
        thrust::fill(thrust::device,
                     buf.begin() + stack_head[0] * n - kThreads * kBlocks * n,
                     buf.begin() + stack_head[0] * n, -1);
        stack_head[0] -= kThreads * kBlocks;
        GenerateKernel<<<kBlocks, kThreads,
                         3 * kThreads * n * sizeof(int8_t)>>>(
            thrust::raw_pointer_cast(temporary_buf.data()),
            thrust::raw_pointer_cast(buf.data()),
            thrust::raw_pointer_cast(stack_head.data()), n,
            thrust::raw_pointer_cast(offsets.data()),
            thrust::raw_pointer_cast(out_edge.data()),
            thrust::raw_pointer_cast(best_td.data()));
        perms += kBlocks * kThreads;
      }
      auto err = cudaStreamSynchronize(0);
      if (err)
        std::cout << cudaGetErrorString(err) << std::endl;
    }
    while (stack_head[0] > 0 &&
           kThreads * kBlocks * n > buf.size() / n - stack_head[0]) {
      int sh = stack_head[0];
      // std::cout << sh << std::endl;
      if (sh < kThreads) {
        BruteForceKernel<<<1, sh, 3 * sh * n * sizeof(int8_t)>>>(
            thrust::raw_pointer_cast(buf.data()), sh, n,
            thrust::raw_pointer_cast(offsets.data()),
            thrust::raw_pointer_cast(out_edge.data()),
            thrust::raw_pointer_cast(best_td.data()));
        perms += sh;
        stack_head[0] -= sh;
      } else if (sh < kBlocks * kThreads) {
        BruteForceKernel<<<sh / kThreads, kThreads,
                           3 * kThreads * n * sizeof(int8_t)>>>(
            thrust::raw_pointer_cast(buf.data()), sh, n,
            thrust::raw_pointer_cast(offsets.data()),
            thrust::raw_pointer_cast(out_edge.data()),
            thrust::raw_pointer_cast(best_td.data()));
        perms += (sh / kThreads) * kThreads;
        stack_head[0] -= (sh / kThreads) * kThreads;
      } else {
        BruteForceKernel<<<kBlocks, kThreads,
                           3 * kThreads * n * sizeof(int8_t)>>>(
            thrust::raw_pointer_cast(buf.data()), sh, n,
            thrust::raw_pointer_cast(offsets.data()),
            thrust::raw_pointer_cast(out_edge.data()),
            thrust::raw_pointer_cast(best_td.data()));
        perms += kBlocks * kThreads;
        stack_head[0] -= kBlocks * kThreads;
      }
      auto err = cudaStreamSynchronize(0);
      if (err)
        std::cout << cudaGetErrorString(err) << std::endl;
    }
  }
  return best_td[0];
}
}  // namespace td
