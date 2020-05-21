// Copyright 2020 GISBDW. All rights reserved.
#pragma once
#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#define HD __host__ __device__
#endif

namespace td {
namespace ext_array_union_find {
template <typename T>
HD T Find(T* uf, T elem) {
  auto root = elem;
  while (uf[root] >= 0)
    root = uf[root];
  while (elem != root) {
    auto prev = uf[elem];
    uf[elem] = root;
    elem = prev;
  }
  return root;
}

template <typename T, typename Offset>
HD void Union(T* uf, T s1, T s2, Offset val_ix) {
  if (s1 == s2)
    return;
  auto new_val = -uf[s1];
  if (new_val <= -uf[s2])
    new_val = -uf[s2] + 1;
  if (uf[val_ix] < new_val)
    uf[val_ix] = new_val;
  uf[s1] = -new_val;
  uf[s2] = s1;
}
}  // namespace ext_array_union_find
}  // namespace td
#ifdef CUDA_ENABLED
#undef HD
#endif
