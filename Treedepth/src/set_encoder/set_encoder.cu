// Copyright 2020 GISBDW. All rights reserved.
#include "set_encoder.hpp"

namespace td {
namespace set_encoder {
std::size_t NChooseK(std::size_t n, std::size_t k) {
  if (k > n)
    return 0;
  if (k > n / 2)
    k = n - k;
  std::size_t ret = 1;
  for (std::size_t i = 1; i <= k; ++i, --n) {
    ret *= n;
    ret /= i;
  }
  return ret;
}

std::size_t Encode(std::vector<bool> const& set) {
  std::size_t ret = 0;
  std::size_t k = 0;
  for (std::size_t i = 0; i < set.size(); ++i)
    if (set[i])
      ret += NChooseK(i, ++k);
  return ret;
}
}  // namespace set_encoder
}  // namespace td
