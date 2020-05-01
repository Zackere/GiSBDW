// Copyright 2020 GISBDW. All rights reserved.

#include "../../src/branch_and_bound/branch_and_bound.hpp"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using ::testing::_;
using ::testing::Return;

namespace {
class MockHeuristic : public td::BranchAndBound::Heuristic {
 public:
  MOCK_METHOD1(Get,
               td::EliminationTree::Result(td::BranchAndBound::Graph const&));
};
class MockLowerBound : public td::BranchAndBound::LowerBound {
 public:
  MOCK_METHOD1(Get, unsigned(td::EliminationTree::Component const&));
};
struct BranchAndBoundTestCase {
  td::BranchAndBound::Graph in;
  td::EliminationTree::Result out;
};
class ParametrizedBranchAndBoundFixture
    : public ::testing::TestWithParam<BranchAndBoundTestCase> {
 public:
  static BranchAndBoundTestCase P3TestCase() {
    BranchAndBoundTestCase ret;
    ret.in = td::BranchAndBound::Graph(3);
    for (int i = 0; i < boost::num_vertices(ret.in) - 1; ++i)
      boost::add_edge(i, i + 1, ret.in);
    ret.out.td_decomp = ret.in;
    ret.out.treedepth = 2;
    ret.out.root = 1;
    return ret;
  }
};
bool CompareBoostGraphs(td::EliminationTree::BoostGraph const& g1,
                        td::EliminationTree::BoostGraph const& g2) {
  if (boost::num_vertices(g1) != boost::num_vertices(g2))
    return false;
  for (int i = 0; i < boost::num_vertices(g1); ++i)
    for (int j = 0; j < i; ++j)
      if (boost::edge(i, j, g1).second != boost::edge(i, j, g2).second)
        return false;
  return true;
}
MATCHER_P(BoostGraphMatcher, g, "") {
  return CompareBoostGraphs(g, arg);
}
}  // namespace

TEST_P(ParametrizedBranchAndBoundFixture, CorrectDecompositionTest) {
  auto& testcase = GetParam();
  MockHeuristic* mock_heuristic = new MockHeuristic();
  MockLowerBound* mock_lower_bound = new MockLowerBound();
  EXPECT_CALL(*mock_heuristic, Get(BoostGraphMatcher(testcase.in)))
      .WillRepeatedly(Return(td::EliminationTree::Result{
          testcase.in, testcase.out.treedepth + 1, testcase.out.root}));
  EXPECT_CALL(*mock_lower_bound, Get(_)).WillRepeatedly(Return(1));
  td::BranchAndBound bnb;
  auto res =
      bnb(testcase.in,
          std::unique_ptr<td::BranchAndBound::LowerBound>(mock_lower_bound),
          std::unique_ptr<td::BranchAndBound::Heuristic>(mock_heuristic));
  EXPECT_EQ(res.root, testcase.out.root);
  EXPECT_EQ(res.treedepth, testcase.out.treedepth);
  EXPECT_TRUE(CompareBoostGraphs(res.td_decomp, testcase.out.td_decomp));
}

INSTANTIATE_TEST_SUITE_P(
    CorrectDataTest,
    ParametrizedBranchAndBoundFixture,
    ::testing::Values(ParametrizedBranchAndBoundFixture::P3TestCase()));
