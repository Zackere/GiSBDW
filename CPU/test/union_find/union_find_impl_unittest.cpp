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

TEST(UnionFindConstructorTest, DoesKeyEqualToOneByDefault) {
  for (int i = 0; i < 10; ++i) {
    auto uf = td::UnionFindImpl(i);
    for (int j = 0; j < i; ++j)
      EXPECT_EQ(1, uf.GetValue(uf.Find(j)));
  }
}

TEST(UnionFindTest, AreElementsInTheSameSetAfterUnion) {
  auto uf = td::UnionFindImpl(11);

  auto id0 = uf.Find(0);
  EXPECT_EQ(id0, uf.Union(id0, uf.Find(5)));
  EXPECT_EQ(id0, uf.Find(0));
  EXPECT_EQ(uf.Find(0), uf.Find(5));

  EXPECT_NE(uf.Find(0), uf.Find(2));
  EXPECT_NE(uf.Find(0), uf.Find(8));
  EXPECT_NE(uf.Find(5), uf.Find(2));
  EXPECT_NE(uf.Find(5), uf.Find(8));

  auto id8 = uf.Find(8);
  EXPECT_EQ(id8, uf.Union(id8, uf.Find(2)));
  EXPECT_EQ(id8, uf.Find(8));
  EXPECT_EQ(uf.Find(2), uf.Find(8));

  EXPECT_NE(uf.Find(0), uf.Find(2));
  EXPECT_NE(uf.Find(0), uf.Find(8));
  EXPECT_NE(uf.Find(5), uf.Find(2));
  EXPECT_NE(uf.Find(5), uf.Find(8));

  EXPECT_EQ(id0, uf.Union(uf.Find(0), uf.Find(2)));
  EXPECT_EQ(id0, uf.Find(0));
  EXPECT_EQ(uf.Find(0), uf.Find(2));
  EXPECT_EQ(uf.Find(0), uf.Find(8));
  EXPECT_EQ(uf.Find(5), uf.Find(2));
  EXPECT_EQ(uf.Find(5), uf.Find(8));
}

TEST(UnionFindTest, GetValueTest1) {
  auto uf = td::UnionFindImpl(12);
  uf.Union(uf.Find(5), uf.Find(7));
  EXPECT_EQ(2, uf.GetValue(uf.Find(5)));
  EXPECT_EQ(2, uf.GetMaxValue());
}

TEST(UnionFindTest, GetValueTest2) {
  auto uf = td::UnionFindImpl(12);
  uf.Union(uf.Find(2), uf.Find(3));
  uf.Union(uf.Find(2), uf.Find(5));
  EXPECT_EQ(2, uf.GetValue(uf.Find(5)));
  EXPECT_EQ(2, uf.GetMaxValue());
}

TEST(UnionFindTest, GetValueTest3) {
  auto uf = td::UnionFindImpl(12);
  uf.Union(uf.Find(2), uf.Find(3));
  uf.Union(uf.Find(5), uf.Find(2));
  EXPECT_EQ(3, uf.GetValue(uf.Find(5)));
  EXPECT_EQ(3, uf.GetMaxValue());
}

TEST(UnionFindTest, GetValueTest4) {
  auto uf = td::UnionFindImpl(12);
  uf.Union(uf.Find(2), uf.Find(3));
  uf.Union(uf.Find(2), uf.Find(5));
  uf.Union(uf.Find(0), uf.Find(5));
  EXPECT_EQ(3, uf.GetValue(uf.Find(5)));
  uf.Union(uf.Find(11), uf.Find(0));
  EXPECT_EQ(4, uf.GetValue(uf.Find(5)));
  EXPECT_EQ(4, uf.GetMaxValue());
}

TEST(UnionFindTest, GetValueTest5) {
  auto uf = td::UnionFindImpl(12);
  uf.Union(uf.Find(3), uf.Find(8));
  uf.Union(uf.Find(2), uf.Find(3));
  uf.Union(uf.Find(2), uf.Find(5));
  EXPECT_EQ(3, uf.GetValue(uf.Find(5)));
  EXPECT_EQ(3, uf.GetMaxValue());
  uf.Union(uf.Find(11), uf.Find(0));
  EXPECT_EQ(2, uf.GetValue(uf.Find(0)));
  EXPECT_EQ(3, uf.GetMaxValue());
  uf.Union(uf.Find(2), uf.Find(11));
  EXPECT_EQ(3, uf.GetValue(uf.Find(5)));
  EXPECT_EQ(3, uf.GetMaxValue());
}

TEST(UnionFindTest, GetValueTest6) {
  // Tree decomposition of P_7
  auto uf = td::UnionFindImpl(7);
  uf.Union(uf.Find(1), uf.Find(0));
  EXPECT_EQ(2, uf.GetMaxValue());
  EXPECT_EQ(2, uf.GetValue(uf.Find(1)));
  uf.Union(uf.Find(1), uf.Find(2));
  EXPECT_EQ(2, uf.GetMaxValue());
  EXPECT_EQ(2, uf.GetValue(uf.Find(1)));
  uf.Union(uf.Find(5), uf.Find(4));
  EXPECT_EQ(2, uf.GetMaxValue());
  EXPECT_EQ(2, uf.GetValue(uf.Find(4)));
  uf.Union(uf.Find(3), uf.Find(1));
  EXPECT_EQ(3, uf.GetMaxValue());
  EXPECT_EQ(3, uf.GetValue(uf.Find(3)));
  uf.Union(uf.Find(5), uf.Find(6));
  EXPECT_EQ(3, uf.GetMaxValue());
  EXPECT_EQ(2, uf.GetValue(uf.Find(6)));
  uf.Union(uf.Find(3), uf.Find(5));
  EXPECT_EQ(3, uf.GetMaxValue());
  EXPECT_EQ(3, uf.GetValue(uf.Find(6)));
  for (int i = 0; i < uf.GetNumberOfElements(); ++i) {
    EXPECT_EQ(3, uf.Find(i));
    EXPECT_EQ(3, uf.GetValue(uf.Find(i)));
  }
}

TEST(UnionFindTest, CloneReturnsSameObject) {
  auto uf = td::UnionFindImpl(10);
  uf.Union(uf.Find(5), uf.Find(7));
  uf.Union(uf.Find(2), uf.Find(7));
  uf.Union(uf.Find(1), uf.Find(9));
  uf.Union(uf.Find(0), uf.Find(3));

  auto uf_clone = uf.Clone();
  auto* uf_p = dynamic_cast<td::UnionFindImpl*>(uf_clone.get());
  ASSERT_NE(nullptr, uf_p);
  ASSERT_EQ(uf.GetNumberOfElements(), uf_p->GetNumberOfElements());
  ASSERT_EQ(uf.GetMaxValue(), uf_p->GetMaxValue());
  for (int i = 0; i < uf.GetNumberOfElements(); ++i) {
    EXPECT_EQ(uf.Find(i), uf_p->Find(i));
    EXPECT_EQ(uf.GetValue(uf.Find(i)), uf_p->GetValue(uf_p->Find(i)));
  }
}
