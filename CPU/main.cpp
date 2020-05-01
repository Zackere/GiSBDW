// Copyright 2020 GISBDW. All rights reserved.

#include <fstream>
#include <iostream>
#include <random>

#include "boost/graph/adjacency_list.hpp"
#include "boost/graph/depth_first_search.hpp"
#include "boost/graph/erdos_renyi_generator.hpp"
#include "boost/graph/graphviz.hpp"
#include "boost/graph/undirected_graph.hpp"

#include "src/dynamic_algorithm/dynamic_algorithm.hpp"

class VertexVisitor : public boost::default_dfs_visitor {
 public:
  template <typename Vertex, typename Graph>
  void discover_vertex(Vertex u, const Graph& g) const {
    std::cout << u << " ";
  }
};

int main() {
  using Graph =
      boost::adjacency_list<boost::mapS, boost::vecS, boost::undirectedS>;
  using ERGen = boost::sorted_erdos_renyi_iterator<std::minstd_rand, Graph>;
  constexpr int n = 18;
  std::minstd_rand rng(0);
  Graph g(ERGen(rng, n, 0.49), ERGen(), n);
  boost::depth_first_search(g, boost::visitor(VertexVisitor()));
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
