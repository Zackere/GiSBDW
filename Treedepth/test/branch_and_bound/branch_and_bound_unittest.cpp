// Copyright 2020 GISBDW. All rights reserved.

#include "../../src/branch_and_bound/branch_and_bound.hpp"

#include "../utils/utils.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using ::testing::_;
using ::testing::Return;

namespace {
class MockHeuristic : public td::BranchAndBound::Heuristic {
 public:
  MOCK_METHOD(td::EliminationTree::Result,
              Get,
              (td::BranchAndBound::Graph const& g),
              (override));
};
class MockLowerBound : public td::BranchAndBound::LowerBound {
 public:
  using variant = std::variant<LowerBoundInfo, TreedepthInfo>;
  MOCK_METHOD(variant,
              Get,
              (td::EliminationTree::Component const& component),
              (override));
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
  EXPECT_CALL(*mock_lower_bound, Get(_))
      .WillRepeatedly(Return(
          td::BranchAndBound::LowerBound::LowerBoundInfo{1, std::nullopt}));
  td::BranchAndBound bnb;
  EXPECT_TRUE(CheckIfTdDecompIsValid(
      testcase.in,
      bnb(testcase.in,
          std::unique_ptr<td::BranchAndBound::LowerBound>(mock_lower_bound),
          std::unique_ptr<td::BranchAndBound::Heuristic>(mock_heuristic)),
      testcase.out));
}

INSTANTIATE_TEST_SUITE_P(
    CorrectDataTest,
    ParametrizedBranchAndBoundFixture,
    ::testing::Values(ParametrizedBranchAndBoundFixture::P3TestCase()));
