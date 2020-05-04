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
  constexpr int n = 22;
  Graph g(n);
  std::set<int8_t> verts;
  for (int i = 0; i < boost::num_vertices(g); ++i)
    verts.insert(i);
  std::minstd_rand rng(time(0));
  g = Graph(ERGen(rng, n, 0.20), ERGen(), n);
#ifdef CUDA_ENABLED
  td::DynamicGPU dgpu;
  dgpu(g);
  auto el = dgpu.GetElimination(verts, boost::num_vertices(g));
  td::EliminationTree et(g);
  for (auto v : el)
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
