// Copyright 2020 GISBDW. All rights reserved.
#pragma once
#include <boost/math/special_functions/binomial.hpp>

namespace td {
size_t NChooseK(unsigned n, unsigned k) {
  if (k > n)
    return 0;
  return static_cast<size_t>(boost::math::binomial_coefficient<double>(n, k));
}
}  // namespace td
