#pragma once

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/erdos_renyi_generator.hpp>
#include <random>

#include "../../src/union_find/array_union_find.hpp"

namespace {
using Graph =
    boost::adjacency_list<boost::mapS, boost::vecS, boost::undirectedS>;
using ERGen = boost::sorted_erdos_renyi_iterator<std::minstd_rand, Graph>;
Graph RandomConnectedGraph(int n, float p, int seed = 0) {
  Graph ret;
  do {
    std::minstd_rand rng(++seed);
    ret = Graph(ERGen(rng, n, p), ERGen(), n);
  } while (boost::connected_components(
               ret, std::vector<int>(boost::num_vertices(ret)).data()) != 1);
  return ret;
}
Graph RandomSparseConnectedGraph(int n, int seed = 0) {
  return RandomConnectedGraph(n, std::log(n) / n, seed);
}
Graph Path(int n) {
  Graph g(n);
  for (std::size_t i = 0; i < n - 1; ++i)
    boost::add_edge(i, i + 1, g);
  return g;
}
Graph Cycle(int n) {
  Graph g = Path(n);
  boost::add_edge(0, n - 1, g);
  return g;
}
Graph Complete(int n) {
  Graph g(n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < i; ++j)
      boost::add_edge(i, j, g);
  return g;
}
Graph SpanningTree(Graph g) {
  Graph ret(boost::num_vertices(g));
  td::ArrayUnionFind<int> uf(boost::num_vertices(g));
  typename boost::graph_traits<
      std::remove_reference_t<decltype(g)>>::out_edge_iterator ei,
      ei_end;
  for (int i = 0; i < boost::num_vertices(g); ++i)
    for (boost::tie(ei, ei_end) = out_edges(i, g); ei != ei_end; ++ei) {
      auto idi = uf.Find(i);
      auto idx = uf.Find(boost::target(*ei, g));
      if (idi != idx) {
        uf.Union(idi, idx);
        boost::add_edge(i, boost::target(*ei, g), ret);
      }
    }
  return ret;
}
Graph Halin(int n) {
  Graph g = RandomSparseConnectedGraph(n);
  std::vector<boost::graph_traits<Graph>::edge_descriptor> p(n);
  g = SpanningTree(g);
  std::vector<boost::graph_traits<Graph>::vertex_descriptor> leaves;
  leaves.reserve(n);
  for (std::size_t i = 0; i < p.size(); ++i)
    if (boost::out_degree(i, g) == 1)
      leaves.emplace_back(i);
  for (std::size_t i = 0; i < leaves.size() - 1; ++i)
    boost::add_edge(leaves[i], leaves[i + 1], g);
  boost::add_edge(leaves[0], leaves[leaves.size() - 1], g);
  return g;
}
Graph ChordalCycle(int n) {
  Graph g = Cycle(n);
  boost::add_edge(0, n / 3, g);
  boost::add_edge(0, 2 * n / 3, g);
  return g;
}
}  // namespace
