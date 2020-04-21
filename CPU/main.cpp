// Copyright 2020 GISBDW. All rights reserved.

#include <fstream>
#include <iostream>
#include <random>

#include "boost/graph/adjacency_list.hpp"
#include "boost/graph/depth_first_search.hpp"
#include "boost/graph/erdos_renyi_generator.hpp"
#include "boost/graph/graphviz.hpp"

class VertexVisitor : public boost::default_dfs_visitor {
 public:
  template <typename Vertex, typename Graph>
  void discover_vertex(Vertex u, const Graph& g) const {
    std::cout << u << " ";
  }
};

int main() {
  using Graph = boost::adjacency_list<>;
  using ERGen = boost::erdos_renyi_iterator<std::minstd_rand, Graph>;
  int n = 25;
  std::minstd_rand rng;
  Graph g(ERGen(rng, n, 0.05), ERGen(), n);
  boost::depth_first_search(g, boost::visitor(VertexVisitor()));
  std::ofstream file("graph.gviz", std::ios_base::trunc);
  boost::write_graphviz(file, g);
  file.close();
  return 0;
}
