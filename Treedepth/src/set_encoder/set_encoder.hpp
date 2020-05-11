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

std::size_t Encode(std::set<int8_t> const& s);

template <typename... Args>
std::size_t Encode(std::map<Args...> const& map) {
  std::size_t ret = 0;
  int i = 0;
  for (auto v : map)
    ret += NChooseK(v.first, ++i);
  return ret;
}

HD std::size_t Encode(int8_t* sorted_set, int k);

HD std::size_t Encode(int8_t* sorted_set, int k, int exclude);

HD void Decode(std::size_t code, int8_t n, int8_t k, int8_t* ret);
}  // namespace set_encoder
}  // namespace td
#undef HD
