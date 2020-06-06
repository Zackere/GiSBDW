// Copyright 2020 GISBDW. All rights reserved.
#pragma once
#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#ifndef HD
#define HD __host__ __device__
#endif
#else
#define HD
#endif

#include <functional>
#include <map>
#include <set>
#include <vector>

namespace td {
namespace set_encoder {
HD std::size_t NChooseK(std::size_t n, std::size_t k);

template <typename Key, typename Allocator>
std::size_t Encode(std::set<Key, std::less<Key>, Allocator> const& s) {
  std::size_t ret = 0;
  std::size_t i = 0;
  for (auto v : s)
    ret += NChooseK(v, ++i);
  return ret;
}

template <typename Key, typename T, typename Allocator>
std::size_t Encode(std::map<Key, T, std::less<Key>, Allocator> const& m) {
  std::size_t ret = 0;
  std::size_t i = 0;
  for (auto v : m)
    ret += NChooseK(v.first, ++i);
  return ret;
}
std::size_t Encode(std::vector<bool> const& set);

template <typename VertexType>
HD std::size_t Encode(VertexType const* sorted_set, std::size_t set_size) {
  std::size_t ret = 0;
  for (std::size_t i = 0; i < set_size; ++i)
    ret += NChooseK(sorted_set[i], i + 1);
  return ret;
}

template <typename VertexType>
HD std::size_t Encode(VertexType const* sorted_set,
                      std::size_t set_size,
                      std::size_t exclude) {
  std::size_t ret = 0;
  std::size_t i = static_cast<std::size_t>(-1);
  while (++i < exclude)
    ret += NChooseK(sorted_set[i], i + 1);
  while (++i < set_size)
    ret += NChooseK(sorted_set[i], i);
  return ret;
}

template <typename VertexType>
HD void Decode(std::size_t nverts,
               std::size_t subset_size,
               std::size_t subset_code,
               VertexType* ret) {
  while (subset_size > 0) {
    auto nk = NChooseK(--nverts, subset_size);
    if (subset_code >= nk) {
      ret[--subset_size] = nverts;
      subset_code -= nk;
    }
  }
}

template <typename Container>
struct DecodeImpl {
  static inline Container Decode(std::size_t nverts,
                                 std::size_t subset_size,
                                 std::size_t subset_code);
};

template <typename Container>
Container Decode(std::size_t nverts,
                 std::size_t subset_size,
                 std::size_t subset_code) {
  return DecodeImpl<Container>::Decode(nverts, subset_size, subset_code);
}

template <typename VertexType>
struct DecodeImpl<std::set<VertexType>> {
  static inline std::set<VertexType> Decode(std::size_t nverts,
                                            std::size_t subset_size,
                                            std::size_t subset_code) {
    std::set<VertexType> ret;
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
};

template <typename VertexType>
struct DecodeImpl<std::vector<VertexType>> {
  static inline std::vector<VertexType> Decode(std::size_t nverts,
                                               std::size_t subset_size,
                                               std::size_t subset_code) {
    std::vector<VertexType> ret(subset_size);
    while (subset_size > 0) {
      auto nk = NChooseK(--nverts, subset_size);
      if (subset_code >= nk) {
        ret[--subset_size] = nverts;
        subset_code -= nk;
      }
    }
    return ret;
  }
};

template <>
struct DecodeImpl<std::vector<bool>> {
  static inline std::vector<bool> Decode(std::size_t nverts,
                                         std::size_t subset_size,
                                         std::size_t subset_code) {
    std::vector<bool> ret(nverts);
    while (subset_size > 0) {
      auto nk = NChooseK(--nverts, subset_size);
      if (subset_code >= nk) {
        ret[nverts] = true;
        subset_code -= nk;
        --subset_size;
      }
    }
    return ret;
  }
};
}  // namespace set_encoder
}  // namespace td
#undef HD
