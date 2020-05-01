// Copyright 2020 GISBDW. All rights reserved.
#pragma once

#include <fstream>
#include <iostream>
#include <limits>
#include <random>

#include "boost/graph/adjacency_list.hpp"
#include "boost/graph/erdos_renyi_generator.hpp"
#include "boost/graph/graphviz.hpp"
#include "src/branch_and_bound/branch_and_bound.hpp"
#include "src/heuristics/highest_degree_heuristic.hpp"
#include "src/lower_bound/basic_lower_bound.hpp"

int main() {
  using Graph =
      boost::adjacency_list<boost::mapS, boost::vecS, boost::undirectedS>;
  using ERGen = boost::sorted_erdos_renyi_iterator<std::minstd_rand, Graph>;
  constexpr int n = 15;
  std::minstd_rand rng;
  Graph g(ERGen(rng, n, 0.5), ERGen(), n);
  td::BranchAndBound bnb;
  auto res = bnb(g, std::make_unique<td::BasicLowerBound>(),
                 std::make_unique<td::HighestDegreeHeuristic>(nullptr));

  std::ofstream file1("graph1.gviz", std::ios_base::trunc);
  boost::write_graphviz(file1, g);
  file1.close();
  std::ofstream file2("graph2.gviz", std::ios_base::trunc);
  boost::write_graphviz(file2, res.td_decomp);
  file2.close();
  return 0;
}
