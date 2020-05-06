// Copyright 2020 GISBDW. All rights reserved.

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

#include "boost/graph/adjacency_list.hpp"
#include "boost/graph/depth_first_search.hpp"
#include "boost/graph/erdos_renyi_generator.hpp"
#include "boost/graph/graphviz.hpp"
#include "boost/program_options.hpp"

namespace po = boost::program_options;

class VertexVisitor : public boost::default_dfs_visitor {
 public:
  template <typename Vertex, typename Graph>
  void discover_vertex(Vertex u, const Graph& g) const {
    std::cout << u << " ";
  }
};

void usage(po::options_description const& description) {
  std::cout << description;
  std::exit(1);
}

int main(int argc, char** argv) {
  po::options_description description("Usage");
  description.add_options()("help", "print this message")(
      "algorithm,a", po::value<std::string>(), "Choose algorithm to run")(
      "input,i", po::value<std::string>(), "path to input graph")(
      "output,o", po::value<std::string>(), "path to output dir");

  // clang-format off
  po::variables_map vm;
  try
  {
  po::store(po::parse_command_line(argc, argv, description), vm);
  }
  catch (po::invalid_command_line_syntax ex)
  {
      std::cout << ex.what() << "\n";
      usage(description);
  }
  // clang-format on
  po::notify(vm);

  if (vm.count("help") == 0 ||
      vm.count("algorithm") == 0 ||
      vm.count("input") == 0 ||
      vm.count("output") == 0)
  {
    usage(description);
  }

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
