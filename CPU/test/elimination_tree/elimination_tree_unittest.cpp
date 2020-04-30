// Copyright 2020 GISBDW. All rights reserved.

#include "../../src/elimination_tree/elimination_tree.hpp"

#include <algorithm>
#include <list>
#include <utility>
#include <vector>

#include "boost/graph/adjacency_list.hpp"
#include "gtest/gtest.h"

struct EliminationTreeTestCase {
  using Graph =
      boost::adjacency_list<boost::mapS, boost::vecS, boost::undirectedS>;
  Graph graph;
  std::vector<td::EliminationTree::VertexType> elimination;
  std::vector<std::list<td::EliminationTree::Component>> components;
};

class ParametrizedEliminationTreeFixture
    : public ::testing::TestWithParam<EliminationTreeTestCase> {
 public:
  static EliminationTreeTestCase C5TestCase() {
    EliminationTreeTestCase tc;
    tc.graph = EliminationTreeTestCase::Graph(5);
    for (uint64_t i = 0; i < boost::num_vertices(tc.graph); ++i)
      boost::add_edge(i, (i + 1) % boost::num_vertices(tc.graph), tc.graph);
    tc.elimination = {0, 3, 2, 4, 1};
    tc.components.reserve(tc.elimination.size() + 1);
    td::EliminationTree::Component c;
    // Eliminate nothing
    tc.components.push_back({});
    c.depth_ = 0;
    c.neighbours_ = {
        {0, {1, 4}}, {1, {0, 2}}, {2, {1, 3}}, {3, {2, 4}}, {4, {0, 3}}};
    tc.components.back().emplace_back(std::move(c));
    // Eliminate 0
    tc.components.push_back({});
    c.depth_ = 1;
    c.neighbours_ = {{1, {2}}, {2, {1, 3}}, {3, {2, 4}}, {4, {3}}};
    tc.components.back().emplace_back(std::move(c));
    // Eliminate 3
    tc.components.push_back({});
    c.depth_ = 2;
    c.neighbours_ = {{4, {}}};
    tc.components.back().emplace_back(std::move(c));
    c.depth_ = 2;
    c.neighbours_ = {{2, {1}}, {1, {2}}};
    tc.components.back().emplace_back(std::move(c));
    // Eliminate 2
    tc.components.push_back({});
    c.depth_ = 2;
    c.neighbours_ = {{4, {}}};
    tc.components.back().emplace_back(std::move(c));
    c.depth_ = 3;
    c.neighbours_ = {{1, {}}};
    tc.components.back().emplace_back(std::move(c));
    // Eliminate 4
    tc.components.push_back({});
    c.depth_ = 3;
    c.neighbours_ = {{1, {}}};
    tc.components.back().emplace_back(std::move(c));
    // Eliminate 1
    tc.components.push_back({});
    return tc;
  }
  static EliminationTreeTestCase TwoCyclesTestCase() {
    EliminationTreeTestCase tc;
    tc.graph = EliminationTreeTestCase::Graph(7);
    boost::add_edge(0, 1, tc.graph);
    boost::add_edge(1, 2, tc.graph);
    boost::add_edge(2, 0, tc.graph);
    boost::add_edge(2, 3, tc.graph);
    boost::add_edge(3, 6, tc.graph);
    boost::add_edge(6, 5, tc.graph);
    boost::add_edge(5, 4, tc.graph);
    boost::add_edge(4, 6, tc.graph);
    tc.elimination = {3, 6, 4, 5, 2, 1, 0};
    tc.components.reserve(tc.elimination.size() + 1);
    td::EliminationTree::Component c;
    // Eliminate nothing
    tc.components.push_back({});
    c.depth_ = 0;
    c.neighbours_ = {{0, {1, 2}}, {1, {0, 2}}, {2, {1, 0, 3}}, {3, {2, 6}},
                     {4, {5, 6}}, {5, {4, 6}}, {6, {3, 4, 5}}};
    tc.components.back().emplace_back(std::move(c));
    // Eliminate 3
    tc.components.push_back({});
    c.depth_ = 1;
    c.neighbours_ = {{0, {1, 2}}, {1, {0, 2}}, {2, {1, 0}}};
    tc.components.back().emplace_back(std::move(c));
    c.depth_ = 1;
    c.neighbours_ = {{4, {5, 6}}, {5, {4, 6}}, {6, {4, 5}}};
    tc.components.back().emplace_back(std::move(c));
    // Eliminate 6
    tc.components.push_back({});
    c.depth_ = 1;
    c.neighbours_ = {{0, {1, 2}}, {1, {0, 2}}, {2, {1, 0}}};
    tc.components.back().emplace_back(std::move(c));
    c.depth_ = 2;
    c.neighbours_ = {{4, {5}}, {5, {4}}};
    tc.components.back().emplace_back(std::move(c));
    // Eliminate 4
    tc.components.push_back({});
    c.depth_ = 1;
    c.neighbours_ = {{0, {1, 2}}, {1, {0, 2}}, {2, {1, 0}}};
    tc.components.back().emplace_back(std::move(c));
    c.depth_ = 3;
    c.neighbours_ = {{5, {}}};
    tc.components.back().emplace_back(std::move(c));
    // Eliminate 5
    tc.components.push_back({});
    c.depth_ = 1;
    c.neighbours_ = {{0, {1, 2}}, {1, {0, 2}}, {2, {1, 0}}};
    tc.components.back().emplace_back(std::move(c));
    // Eliminate 2
    tc.components.push_back({});
    c.depth_ = 2;
    c.neighbours_ = {{0, {1}}, {1, {0}}};
    tc.components.back().emplace_back(std::move(c));
    // Eliminate 1
    tc.components.push_back({});
    c.depth_ = 3;
    c.neighbours_ = {{0, {}}};
    tc.components.back().emplace_back(std::move(c));
    // Eliminate 0
    tc.components.push_back({});
    return tc;
  }
  static EliminationTreeTestCase PathSimpleTestCase() {
    EliminationTreeTestCase tc;
    tc.graph = EliminationTreeTestCase::Graph(4);
    for (uint64_t i = 0; i < boost::num_vertices(tc.graph) - 1; ++i)
      boost::add_edge(i, i + 1, tc.graph);
    tc.elimination = {3, 2, 1, 0};
    tc.components.reserve(tc.elimination.size() + 1);
    td::EliminationTree::Component c;
    // Eliminate nothing
    tc.components.push_back({});
    c.depth_ = 0;
    c.neighbours_ = {{0, {1}}, {1, {0, 2}}, {2, {1, 3}}, {3, {2}}};
    tc.components.back().emplace_back(std::move(c));
    // Eliminate 3
    tc.components.push_back({});
    c.depth_ = 1;
    c.neighbours_ = {{0, {1}}, {1, {0, 2}}, {2, {1}}};
    tc.components.back().emplace_back(std::move(c));
    // Eliminate 2
    tc.components.push_back({});
    c.depth_ = 2;
    c.neighbours_ = {{0, {1}}, {1, {0}}};
    tc.components.back().emplace_back(std::move(c));
    // Eliminate 1
    tc.components.push_back({});
    c.depth_ = 3;
    c.neighbours_ = {{0, {}}};
    tc.components.back().emplace_back(std::move(c));
    // Eliminate 0
    tc.components.push_back({});
    return tc;
  }
  static EliminationTreeTestCase PathOptimalTestCase() {
    EliminationTreeTestCase tc;
    tc.graph = EliminationTreeTestCase::Graph(7);
    for (uint64_t i = 0; i < boost::num_vertices(tc.graph) - 1; ++i)
      boost::add_edge(i, i + 1, tc.graph);
    tc.elimination = {3, 1, 5, 0, 2, 4, 6};
    tc.components.reserve(tc.elimination.size() + 1);
    td::EliminationTree::Component c;
    // Eliminate nothing
    tc.components.push_back({});
    c.depth_ = 0;
    c.neighbours_ = {{0, {1}},    {1, {0, 2}}, {2, {1, 3}}, {3, {2, 4}},
                     {4, {3, 5}}, {5, {4, 6}}, {6, {5}}};
    tc.components.back().emplace_back(std::move(c));
    // Eliminate 3
    tc.components.push_back({});
    c.depth_ = 1;
    c.neighbours_ = {{0, {1}}, {1, {0, 2}}, {2, {1}}};
    tc.components.back().emplace_back(std::move(c));
    c.depth_ = 1;
    c.neighbours_ = {{4, {5}}, {5, {4, 6}}, {6, {5}}};
    tc.components.back().emplace_back(std::move(c));
    // Eliminate 1
    tc.components.push_back({});
    c.depth_ = 2;
    c.neighbours_ = {{0, {}}};
    tc.components.back().emplace_back(std::move(c));
    c.depth_ = 2;
    c.neighbours_ = {{2, {}}};
    tc.components.back().emplace_back(std::move(c));
    c.depth_ = 1;
    c.neighbours_ = {{4, {5}}, {5, {4, 6}}, {6, {5}}};
    tc.components.back().emplace_back(std::move(c));
    // Eliminate 5
    tc.components.push_back({});
    c.depth_ = 2;
    c.neighbours_ = {{0, {}}};
    tc.components.back().emplace_back(std::move(c));
    c.depth_ = 2;
    c.neighbours_ = {{2, {}}};
    tc.components.back().emplace_back(std::move(c));
    c.depth_ = 2;
    c.neighbours_ = {{4, {}}};
    tc.components.back().emplace_back(std::move(c));
    c.depth_ = 2;
    c.neighbours_ = {{6, {}}};
    tc.components.back().emplace_back(std::move(c));
    // Eliminate 0
    tc.components.push_back({});
    c.depth_ = 2;
    c.neighbours_ = {{2, {}}};
    tc.components.back().emplace_back(std::move(c));
    c.depth_ = 2;
    c.neighbours_ = {{4, {}}};
    tc.components.back().emplace_back(std::move(c));
    c.depth_ = 2;
    c.neighbours_ = {{6, {}}};
    tc.components.back().emplace_back(std::move(c));
    // Eliminate 2
    tc.components.push_back({});
    c.depth_ = 2;
    c.neighbours_ = {{4, {}}};
    tc.components.back().emplace_back(std::move(c));
    c.depth_ = 2;
    c.neighbours_ = {{6, {}}};
    tc.components.back().emplace_back(std::move(c));
    // Eliminate 4
    tc.components.push_back({});
    c.depth_ = 2;
    c.neighbours_ = {{6, {}}};
    tc.components.back().emplace_back(std::move(c));
    // Eliminate 6
    tc.components.push_back({});
    return tc;
  }
};

TEST_P(ParametrizedEliminationTreeFixture, CorrectEliminationTest) {
  auto& testcase = GetParam();
  td::EliminationTree et(testcase.graph);
  for (int i = 0; i < testcase.elimination.size(); ++i) {
    EXPECT_EQ(testcase.components[i].size(),
              std::distance(et.ComponentsBegin(), et.ComponentsEnd()));
    std::for_each(
        et.ComponentsBegin(), et.ComponentsEnd(), [&](auto const& component) {
          EXPECT_NE(std::find(std::begin(testcase.components[i]),
                              std::end(testcase.components[i]), component),
                    std::end(testcase.components[i]));
        });
    et.Eliminate(testcase.elimination[i]);
  }
  EXPECT_EQ(testcase.components[testcase.elimination.size()].size(),
            std::distance(et.ComponentsBegin(), et.ComponentsEnd()));
}

TEST_P(ParametrizedEliminationTreeFixture, CorrectMergeTest) {
  auto& testcase = GetParam();
  td::EliminationTree et(testcase.graph);
  for (auto v : testcase.elimination)
    et.Eliminate(v);
  for (int i = testcase.elimination.size(); i > 0; --i) {
    EXPECT_EQ(testcase.components[i].size(),
              std::distance(et.ComponentsBegin(), et.ComponentsEnd()));
    std::for_each(
        et.ComponentsBegin(), et.ComponentsEnd(), [&](auto const& component) {
          EXPECT_NE(std::find(std::begin(testcase.components[i]),
                              std::end(testcase.components[i]), component),
                    std::end(testcase.components[i]));
        });
    et.Merge();
  }
  EXPECT_EQ(testcase.components[0].size(),
            std::distance(et.ComponentsBegin(), et.ComponentsEnd()));
  std::for_each(
      et.ComponentsBegin(), et.ComponentsEnd(), [&](auto const& component) {
        EXPECT_NE(std::find(std::begin(testcase.components[0]),
                            std::end(testcase.components[0]), component),
                  std::end(testcase.components[0]));
      });
}

INSTANTIATE_TEST_SUITE_P(
    EliminationTest,
    ParametrizedEliminationTreeFixture,
    ::testing::Values(ParametrizedEliminationTreeFixture::C5TestCase(),
                      ParametrizedEliminationTreeFixture::PathOptimalTestCase(),
                      ParametrizedEliminationTreeFixture::PathSimpleTestCase(),
                      ParametrizedEliminationTreeFixture::TwoCyclesTestCase()));
INSTANTIATE_TEST_SUITE_P(
    MergeTest,
    ParametrizedEliminationTreeFixture,
    ::testing::Values(ParametrizedEliminationTreeFixture::C5TestCase(),
                      ParametrizedEliminationTreeFixture::PathOptimalTestCase(),
                      ParametrizedEliminationTreeFixture::PathSimpleTestCase(),
                      ParametrizedEliminationTreeFixture::TwoCyclesTestCase()));
