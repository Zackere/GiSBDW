// Copyright 2020 GISBDW. All rights reserved.
#pragma once
#define HD
#ifdef CUDA_ENABLED
#include "cuda_runtime.h"
#define HD __host__ __device__
#endif

#include <map>
#include <set>

namespace td {
namespace set_encoder {
HD std::size_t NChooseK(int n, int k);

template <typename Key, typename Allocator>
std::size_t Encode(std::set<Key, std::less<Key>, Allocator> const& s) {
  std::size_t ret = 0;
  int i = 0;
  for (auto v : s)
    ret += NChooseK(v, ++i);
  return ret;
}

template <typename Key, typename T, typename Allocator>
std::size_t Encode(std::map<Key, T, std::less<Key>, Allocator> const& map) {
  std::size_t ret = 0;
  int i = 0;
  for (auto v : map)
    ret += NChooseK(v.first, ++i);
  return ret;
}

HD std::size_t Encode(int8_t* sorted_set, int k);

HD std::size_t Encode(int8_t* sorted_set, int k, int exclude);

HD void Decode(std::size_t code, int8_t n, int8_t k, int8_t* ret);

template <typename... Args>
std::set<Args...> Decode(std::size_t nverts,
                         std::size_t subset_size,
                         std::size_t subset_code) {
  std::set<Args...> ret;
  while (subset_size > 0) {
    auto nk = NChooseK(--nverts, subset_size);
    if (subset_code >= nk) {
      ret.insert(nverts);
      subset_code -= nk;
      --subset_size;
    }
  }
  return ret;
}
}  // namespace set_encoder
}  // namespace td
#undef HD
