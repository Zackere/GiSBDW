// Copyright 2020 GISBDW. All rights reserved.

#include "../../src/union_find/union_find_impl.hpp"

#include "gtest/gtest.h"

TEST(UnionFindConstructorTest, IsSizeCorrect) {
  for (int i = 0; i < 10; ++i) {
    auto uf = td::UnionFindImpl(i);
    ASSERT_EQ(i, uf.GetNumberOfElements());
  }
}

TEST(UnionFindConstructorTest, DoesFindReturnCorrectSetId) {
  for (int i = 0; i < 10; ++i) {
    auto uf = td::UnionFindImpl(i);
    for (int j = 0; j < i; ++j)
      ASSERT_EQ(j, uf.Find(j));
  }
}

TEST(UnionFindTest, AreElementsInTheSameSetAfterUnion) {
  auto uf = td::UnionFindImpl(10);

  auto id0 = uf.Find(0);
  ASSERT_EQ(id0, uf.Union(id0, uf.Find(5)));
  ASSERT_EQ(uf.Find(0), uf.Find(5));

  ASSERT_NE(uf.Find(0), uf.Find(2));
  ASSERT_NE(uf.Find(0), uf.Find(8));
  ASSERT_NE(uf.Find(5), uf.Find(2));
  ASSERT_NE(uf.Find(5), uf.Find(8));

  auto id8 = uf.Find(8);
  ASSERT_EQ(id8, uf.Union(id8, uf.Find(2)));
  ASSERT_EQ(uf.Find(2), uf.Find(8));

  ASSERT_NE(uf.Find(0), uf.Find(2));
  ASSERT_NE(uf.Find(0), uf.Find(8));
  ASSERT_NE(uf.Find(5), uf.Find(2));
  ASSERT_NE(uf.Find(5), uf.Find(8));

  ASSERT_EQ(id0, uf.Union(uf.Find(0), uf.Find(2)));
  ASSERT_EQ(uf.Find(0), uf.Find(2));
  ASSERT_EQ(uf.Find(0), uf.Find(8));
  ASSERT_EQ(uf.Find(5), uf.Find(2));
  ASSERT_EQ(uf.Find(5), uf.Find(8));
}
