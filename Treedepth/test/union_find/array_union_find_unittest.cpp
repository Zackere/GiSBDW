// Copyright 2020 GISBDW. All rights reserved.

#include "../../src/union_find/array_union_find.hpp"

#include "gtest/gtest.h"

TEST(ArrayUnionFindConstructorTest, DoesFindReturnCorrectSetId) {
  for (int i = 0; i < 10; ++i) {
    td::ArrayUnionFind<int> uf(i);
    for (int j = 0; j < i; ++j)
      EXPECT_EQ(j, uf.Find(j));
  }
}

TEST(ArrayUnionFindConstructorTest, DoesKeyEqualToOneByDefault) {
  for (int i = 0; i < 10; ++i) {
    td::ArrayUnionFind<int> uf(i);
    EXPECT_EQ(1, uf.GetMaxValue());
    for (int j = 0; j < i; ++j)
      EXPECT_EQ(1, uf.GetValue(uf.Find(j)));
  }
}

TEST(ArrayUnionFindTest, AreElementsInTheSameSetAfterUnion) {
  td::ArrayUnionFind<int> uf(11);

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

TEST(ArrayUnionFindTest, GetValueTest1) {
  td::ArrayUnionFind<int> uf(12);
  uf.Union(uf.Find(5), uf.Find(7));
  EXPECT_EQ(2, uf.GetValue(uf.Find(5)));
  EXPECT_EQ(2, uf.GetMaxValue());
}

TEST(ArrayUnionFindTest, GetValueTest2) {
  td::ArrayUnionFind<int> uf(12);
  uf.Union(uf.Find(2), uf.Find(3));
  uf.Union(uf.Find(2), uf.Find(5));
  EXPECT_EQ(2, uf.GetValue(uf.Find(5)));
  EXPECT_EQ(2, uf.GetMaxValue());
}

TEST(ArrayUnionFindTest, GetValueTest3) {
  td::ArrayUnionFind<int> uf(12);
  uf.Union(uf.Find(2), uf.Find(3));
  uf.Union(uf.Find(5), uf.Find(2));
  EXPECT_EQ(3, uf.GetValue(uf.Find(5)));
  EXPECT_EQ(3, uf.GetMaxValue());
}

TEST(ArrayUnionFindTest, GetValueTest4) {
  td::ArrayUnionFind<int> uf(12);
  uf.Union(uf.Find(2), uf.Find(3));
  uf.Union(uf.Find(2), uf.Find(5));
  uf.Union(uf.Find(0), uf.Find(5));
  EXPECT_EQ(3, uf.GetValue(uf.Find(5)));
  uf.Union(uf.Find(11), uf.Find(0));
  EXPECT_EQ(4, uf.GetValue(uf.Find(5)));
  EXPECT_EQ(4, uf.GetMaxValue());
}

TEST(ArrayUnionFindTest, GetValueTest5) {
  td::ArrayUnionFind<int> uf(12);
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

TEST(ArrayUnionFindTest, GetValueTest6) {
  // Tree decomposition of P_7
  td::ArrayUnionFind<int> uf(7);
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
  for (int i = 0; i < 7; ++i) {
    EXPECT_EQ(3, uf.Find(i));
    EXPECT_EQ(3, uf.GetValue(uf.Find(i)));
  }
}

TEST(ArrayUnionFindTest, GetValueTest7) {
  td::ArrayUnionFind<int> uf(15);
  uf.Union(4, 9);
  uf.Union(7, 2);
  uf.Union(7, 5);
  uf.Union(11, 1);
  uf.Union(11, 13);
  EXPECT_EQ(uf.GetMaxValue(), 2);
  uf.Union(3, 4);
  uf.Union(3, 7);
  uf.Union(3, 10);
  uf.Union(8, 6);
  uf.Union(8, 11);
  EXPECT_EQ(uf.GetMaxValue(), 3);
  uf.Union(12, 3);
  uf.Union(12, 8);
  EXPECT_EQ(uf.GetMaxValue(), 4);
  uf.Union(0, 12);
  uf.Union(0, 14);
  EXPECT_EQ(uf.GetMaxValue(), 5);
}

TEST(ArrayUnionFindTest, CopyConstructorTest) {
  td::ArrayUnionFind<int> uf(10);
  uf.Union(uf.Find(5), uf.Find(7));
  uf.Union(uf.Find(2), uf.Find(7));
  uf.Union(uf.Find(1), uf.Find(9));
  uf.Union(uf.Find(0), uf.Find(3));

  auto uf_clone = uf;
  ASSERT_EQ(uf.GetMaxValue(), uf.GetMaxValue());
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(uf.Find(i), uf_clone.Find(i));
    EXPECT_EQ(uf.GetValue(uf.Find(i)), uf_clone.GetValue(uf_clone.Find(i)));
  }
}
