// Copyright 2020 GISBDW. All rights reserved.

#include <fstream>
#include <random>

#include "boost/graph/adjacency_list.hpp"
#include "boost/graph/erdos_renyi_generator.hpp"
#include "boost/graph/graphviz.hpp"
#include "boost/graph/undirected_graph.hpp"

#include "src/dynamic_algorithm/dynamic_algorithm.hpp"

int main() {
  using Graph =
      boost::adjacency_list<boost::mapS, boost::vecS, boost::undirectedS>;
  using ERGen = boost::sorted_erdos_renyi_iterator<std::minstd_rand, Graph>;
  constexpr int n = 10;
  std::minstd_rand rng(0);
  Graph g(ERGen(rng, n, 0.2), ERGen(), n);
  std::cout << std::endl;
  std::ofstream file("graph.gviz", std::ios_base::trunc);
  boost::write_graphviz(file, g);
  file.close();
  td::DynamicAlgorithm<int> foo;
  foo.Test();
  std::cout << "Is undirected? -> " << boost::is_undirected(g) << "\n";
  auto td = foo.ComputeTreeDepth(g);
  std::cout << "Tree depth -> " << td << "\n";
  return 0;
}
