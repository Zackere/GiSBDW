// Copyright 2020 GISBDW. All rights reserved.

#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>

#include "boost/graph/adjacency_list.hpp"
#include "boost/graph/depth_first_search.hpp"
#include "boost/graph/erdos_renyi_generator.hpp"
#include "boost/graph/graphviz.hpp"
#include "src/branch_and_bound/branch_and_bound.hpp"

struct VertexVisitor : public boost::default_dfs_visitor {
  template <typename Vertex, typename Graph>
  void discover_vertex(Vertex u, Graph const& g) const {
    std::cout << u << ' ';
  }
};

int main() {
  using Graph =
      boost::adjacency_list<boost::mapS, boost::vecS, boost::undirectedS>;
  using ERGen = boost::sorted_erdos_renyi_iterator<std::minstd_rand, Graph>;
  constexpr int n = 15;
  using namespace std::chrono;
  std::minstd_rand rng(
      duration_cast<seconds>(system_clock::now().time_since_epoch()).count());
  Graph g(ERGen(rng, n, 0.2), ERGen(), n);
  boost::depth_first_search(g, boost::visitor(VertexVisitor()));
  std::ofstream file("graph.gviz", std::ios_base::trunc);
  boost::write_graphviz(file, g);
  file.close();
  return 0;
}
