// Copyright 2020 GISBDW. All rights reserved.

#include "../../src/set_encoder/set_encoder.hpp"

#include <algorithm>
#include <list>
#include <set>
#include <vector>

#include "gtest/gtest.h"

namespace {
class SECF : public ::testing::TestWithParam<std::size_t> {};
}  // namespace

TEST_P(SECF, NChooseKTest) {
  auto n = GetParam();
  EXPECT_EQ(td::set_encoder::NChooseK(n, 0), 1);
  EXPECT_EQ(td::set_encoder::NChooseK(n, n), 1);
  for (std::size_t i = 1; i < n; ++i) {
    EXPECT_EQ(td::set_encoder::NChooseK(n, i),
              td::set_encoder::NChooseK(n - 1, i - 1) +
                  td::set_encoder::NChooseK(n - 1, i));
  }
}

TEST_P(SECF, STDSetEncodeDecode) {
  auto n = GetParam();
  for (std::size_t i = 0; i <= n; ++i) {
    std::list<std::set<unsigned>> sets;
    for (std::size_t j = 0; j < td::set_encoder::NChooseK(n, i); ++j) {
      auto set = td::set_encoder::Decode<std::set<unsigned>>(n, i, j);
      for (auto v : set)
        EXPECT_TRUE(0 <= v && v < n);
      EXPECT_EQ(set.size(), i);
      EXPECT_EQ(td::set_encoder::Encode(set), j);
      EXPECT_EQ(std::find(std::begin(sets), std::end(sets), set),
                std::end(sets));
      sets.push_back(std::move(set));
    }
  }
}

TEST_P(SECF, STDVectorEncodeDecode) {
  auto n = GetParam();
  for (std::size_t i = 0; i <= n; ++i) {
    std::list<std::vector<unsigned>> sets;
    for (std::size_t j = 0; j < td::set_encoder::NChooseK(n, i); ++j) {
      auto vec = td::set_encoder::Decode<std::vector<unsigned>>(n, i, j);
      for (auto v : vec)
        EXPECT_TRUE(0 <= v && v < n);
      EXPECT_EQ(vec.size(), i);
      EXPECT_EQ(td::set_encoder::Encode(vec.data(), vec.size()), j);
      EXPECT_EQ(std::find(std::begin(sets), std::end(sets), vec),
                std::end(sets));
      sets.push_back(std::move(vec));
    }
  }
}

TEST_P(SECF, PointerEncodeDecode) {
  auto n = GetParam();
  for (std::size_t i = 0; i <= n; ++i) {
    for (std::size_t j = 0; j < td::set_encoder::NChooseK(n, i); ++j) {
      std::vector<unsigned> vec1(i);
      td::set_encoder::Decode(n, i, j, vec1.data());
      auto vec2 = td::set_encoder::Decode<std::vector<unsigned>>(n, i, j);
      EXPECT_EQ(vec1.size(), i);
      EXPECT_EQ(vec2.size(), i);
      EXPECT_EQ(vec1, vec2);
      for (auto v : vec1)
        EXPECT_TRUE(0 <= v && v < n);
    }
  }
}

TEST_P(SECF, EncodeExluded) {
  auto n = GetParam();
  for (std::size_t i = 0; i <= n; ++i) {
    for (std::size_t j = 0; j < td::set_encoder::NChooseK(n, i); ++j) {
      auto set = td::set_encoder::Decode<std::set<unsigned>>(n, i, j);
      auto vec = td::set_encoder::Decode<std::vector<unsigned>>(n, i, j);
      EXPECT_EQ(set.size(), i);
      EXPECT_EQ(vec.size(), i);
      EXPECT_TRUE(std::equal(std::begin(set), std::end(set), std::begin(vec),
                             std::end(vec)));
      for (auto v : vec)
        EXPECT_TRUE(0 <= v && v < n);
      for (std::size_t i = 0; i < vec.size(); ++i) {
        set.erase(vec[i]);
        EXPECT_EQ(td::set_encoder::Encode(vec.data(), vec.size(), i),
                  td::set_encoder::Encode(set));
        set.insert(vec[i]);
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    SetSizes,
    SECF,
    ::testing::Values(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15));
