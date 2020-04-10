// Copyright 2020 GISBDW. All rights reserved.

#include "../../src/union_find/union_find_impl.hpp"

#include "gtest/gtest.h"

TEST(UnionFindConstructorTest, IsSizeCorrect) {
  for (int i = 0; i < 10; ++i) {
    auto uf = td::UnionFindImpl(i);
    EXPECT_EQ(i, uf.GetNumberOfElements());
  }
}

TEST(UnionFindConstructorTest, DoesFindReturnCorrectSetId) {
  for (int i = 0; i < 10; ++i) {
    auto uf = td::UnionFindImpl(i);
    for (int j = 0; j < i; ++j)
      EXPECT_EQ(j, uf.Find(j));
  }
}

TEST(UnionFindTest, AreElementsInTheSameSetAfterUnion) {
  auto uf = td::UnionFindImpl(11);

  auto id0 = uf.Find(0);
  EXPECT_EQ(id0, uf.Union(id0, uf.Find(5)));
  EXPECT_EQ(uf.Find(0), uf.Find(5));

  EXPECT_NE(uf.Find(0), uf.Find(2));
  EXPECT_NE(uf.Find(0), uf.Find(8));
  EXPECT_NE(uf.Find(5), uf.Find(2));
  EXPECT_NE(uf.Find(5), uf.Find(8));

  auto id8 = uf.Find(8);
  EXPECT_EQ(id8, uf.Union(id8, uf.Find(2)));
  EXPECT_EQ(uf.Find(2), uf.Find(8));

  EXPECT_NE(uf.Find(0), uf.Find(2));
  EXPECT_NE(uf.Find(0), uf.Find(8));
  EXPECT_NE(uf.Find(5), uf.Find(2));
  EXPECT_NE(uf.Find(5), uf.Find(8));

  EXPECT_EQ(id0, uf.Union(uf.Find(0), uf.Find(2)));
  EXPECT_EQ(uf.Find(0), uf.Find(2));
  EXPECT_EQ(uf.Find(0), uf.Find(8));
  EXPECT_EQ(uf.Find(5), uf.Find(2));
  EXPECT_EQ(uf.Find(5), uf.Find(8));
}

TEST(UnionFindTest, GetSetValueTest) {
  auto uf = td::UnionFindImpl(12);
  uf.Union(uf.Find(5), uf.Find(7));
  uf.Union(uf.Find(2), uf.Find(7));
  uf.Union(uf.Find(0), uf.Find(9));

  uf.SetValue(uf.Find(5), 123);
  EXPECT_EQ(uf.GetValue(uf.Find(5)), 123);
  EXPECT_EQ(uf.GetMaxValue(), 123);

  uf.SetValue(uf.Find(5), 12);
  EXPECT_EQ(uf.GetValue(uf.Find(5)), 12);
  EXPECT_EQ(uf.GetMaxValue(), 12);

  uf.SetValue(uf.Find(0), 15);
  EXPECT_EQ(uf.GetValue(uf.Find(0)), 15);
  EXPECT_EQ(uf.GetMaxValue(), 15);

  uf.SetValue(uf.Find(0), -15);
  EXPECT_EQ(uf.GetValue(uf.Find(0)), -15);
  EXPECT_EQ(uf.GetMaxValue(), -15);

  uf.SetValue(uf.Find(5), 123);
  EXPECT_EQ(uf.GetValue(uf.Find(5)), 123);
  EXPECT_EQ(uf.GetMaxValue(), 123);

  EXPECT_EQ(uf.GetValue(uf.Find(0)), -15);
}

TEST(UnionFindTest, CloneReturnsSameObject) {
  auto uf = td::UnionFindImpl(10);
  uf.Union(uf.Find(5), uf.Find(7));
  uf.Union(uf.Find(2), uf.Find(7));
  uf.Union(uf.Find(1), uf.Find(9));
  uf.SetValue(uf.Find(5), 123);
  uf.SetValue(uf.Find(8), 321);
  uf.SetValue(uf.Find(0), -112);

  auto uf_clone = uf.Clone();
  auto* uf_p = dynamic_cast<td::UnionFindImpl*>(uf_clone.get());
  ASSERT_NE(nullptr, uf_p);
  ASSERT_EQ(uf.GetNumberOfElements(), uf_p->GetNumberOfElements());
  ASSERT_EQ(uf.GetMaxValue(), uf_p->GetMaxValue());
  for (int i = 0; i < uf.GetNumberOfElements(); ++i) {
    EXPECT_EQ(uf.Find(i), uf_p->Find(i));
    EXPECT_EQ(uf.GetValue(i), uf_p->GetValue(i));
  }
}