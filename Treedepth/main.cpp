// Copyright 2020 GISBDW. All rights reserved.

#include <fstream>
#include <random>

#include "boost/graph/adjacency_list.hpp"
#include "boost/graph/erdos_renyi_generator.hpp"
#include "boost/graph/graphviz.hpp"
#include "src/branch_and_bound/branch_and_bound.hpp"
#include "src/dynamic_gpu/dynamic_gpu.hpp"
#include "src/heuristics/highest_degree_heuristic.hpp"
#include "src/lower_bound/basic_lower_bound.hpp"

int main() {
  using Graph =
      boost::adjacency_list<boost::mapS, boost::vecS, boost::undirectedS>;
  using ERGen = boost::sorted_erdos_renyi_iterator<std::minstd_rand, Graph>;
  int n = 5;
  Graph g;
  td::EliminationTree::Result res;
  while (true) {
    try {
      std::minstd_rand rng(time(0));
      g = Graph(ERGen(rng, n, 0.5), ERGen(), n);
      td::BranchAndBound bnb;
      res = bnb(g, std::make_unique<td::BasicLowerBound>(),
                std::make_unique<td::HighestDegreeHeuristic>(nullptr));
      break;
    } catch (...) {
    }
  }

  std::ofstream file1("graph1.gviz", std::ios_base::trunc);
  boost::write_graphviz(file1, g);
  file1.close();
  std::ofstream file2("graph2.gviz", std::ios_base::trunc);
  boost::write_graphviz(file2, res.td_decomp);
  file2.close();
  td::DynamicGPU dgpu;
  dgpu(Graph(1));
  return 0;
}
