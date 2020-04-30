// Copyright 2020 GISBDW. All rights reserved.

#include "../../src/elimination_tree/elimination_tree.hpp"

#include <algorithm>
#include <list>
#include <utility>
#include <vector>

#include "boost/graph/adjacency_list.hpp"
#include "gtest/gtest.h"

class EliminationTreeFixture : public ::testing::Test {
 protected:
  using Graph =
      boost::adjacency_list<boost::mapS, boost::vecS, boost::undirectedS>;
  void SetUp() override { AddC5TestCase(); }
  struct TestCase {
    Graph graph;
    std::vector<td::EliminationTree::VertexType> elimination;
    std::vector<std::list<td::EliminationTree::Component>> components;
  };
  std::list<TestCase> data_;

 private:
  void AddC5TestCase() {
    TestCase tc;
    tc.graph = Graph(5);
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

    data_.emplace_back(std::move(tc));
  }
};

TEST_F(EliminationTreeFixture, EliminationTest) {
  for (auto& testcase : data_) {
    td::EliminationTree et(testcase.graph);
    for (int i = 0; i < testcase.elimination.size(); ++i) {
      EXPECT_EQ(testcase.components[i].size(),
                std::distance(et.ComponentsBegin(), et.ComponentsEnd()));
      et.Eliminate(testcase.elimination[i]);
    }
    EXPECT_EQ(et.ComponentsBegin(), et.ComponentsEnd());
  }
}
