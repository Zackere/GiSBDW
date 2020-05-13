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
  constexpr int n = 25;
  Graph g(n);
  std::minstd_rand rng(time(0));
  do {
    g = Graph(ERGen(rng, n, 0.20), ERGen(), n);
  } while (
      boost::connected_components(
          g, std::vector<decltype(g)::vertex_descriptor>(boost::num_vertices(g))
                 .data()) != 1);
#ifdef CUDA_ENABLED
  td::DynamicGPU dgpu;
  if (dgpu.GetMaxIterations(boost::num_vertices(g), boost::num_edges(g), 0) !=
      boost::num_vertices(g) + 1) {
    std::cout << "Not enough mem\n";
    return 0;
  }
  dgpu(g);
  td::EliminationTree et(g);
  for (auto v : dgpu.GetElimination<int>(boost::num_vertices(g),
                                         boost::num_vertices(g), 0))
    et.Eliminate(v);
  auto res = et.Decompose();
  std::ofstream file1("graph1.gviz", std::ios_base::trunc);
  boost::write_graphviz(file1, g);
  file1.close();
  std::ofstream file2("graph2.gviz", std::ios_base::trunc);
  boost::write_graphviz(file2, res.td_decomp);
  file2.close();
#endif
  return 0;
}
