// Copyright 2020 GISBDW. All rights reserved.
#include "set_encoder.hpp"

namespace td {
namespace set_encoder {
std::size_t NChooseK(int n, int k) {
  if (k < 0 || k > n)
    return 0;
  if (k > n - k)
    k = n - k;
  std::size_t ret = 1;
  for (int i = 1; i <= k; ++i, --n) {
    ret *= n;
    ret /= i;
  }
  return ret;
}

std::size_t Encode(int8_t* sorted_set, int k) {
  std::size_t ret = 0;
  for (int i = 0; i < k; ++i)
    ret += NChooseK(sorted_set[i], i + 1);
  return ret;
}

std::size_t Encode(int8_t* sorted_set, int k, int exclude) {
  std::size_t ret = 0;
  int i = -1;
  while (++i < exclude)
    ret += NChooseK(sorted_set[i], i + 1);
  while (++i < k)
    ret += NChooseK(sorted_set[i], i);
  return ret;
}

void Decode(std::size_t code, int8_t n, int8_t k, int8_t* ret) {
  while (k > 0) {
    auto nk = NChooseK(--n, k);
    if (code >= nk) {
      ret[--k] = n;
      code -= nk;
    }
  }
}
}  // namespace set_encoder
}  // namespace td
