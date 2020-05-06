// Copyright 2020 GISBDW. All rights reserved.
// clang-format off

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <filesystem>

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

bool PathExists(std::string const& path)
{
    if (!std::filesystem::exists(path))
    {
        std::cerr << path << " does not exist.\n";
        return false;
    }
    return true;
}

int main(int argc, char** argv) {

    std::string algorithmType;
    std::string outputDir;
    std::vector<std::string> graphsPaths;
  po::options_description description("Usage");
  description.add_options()
      ("help", "print this message")(
      "algorithm,a", po::value<std::string>(&algorithmType)->required(),
          "Choose algorithm to run.\n"
          "Possible args:\n"
          "bnb - for branch and bound algorithm\n"
          "dyn - for dynamic algorithm\n"
          "hyb - for hybrid algorithm\n")(
      "input,i", po::value<std::vector<std::string>>(&graphsPaths)->required(), "path to input graph")(
      "output,o", po::value<std::string>(&outputDir)->required(), "path to output dir");

  po::positional_options_description positionalArgs;
  positionalArgs.add("input", -1);
  po::variables_map vm;
  try
  {
    po::store(po::command_line_parser(argc, argv).
        options(description).positional(positionalArgs).run(), vm);
  if (vm.count("help")){
    usage(description);
  }
    po::notify(vm);
  }
  catch (po::error& ex)
  {
      std::cerr << ex.what() << "\n";
      usage(description);
  }

  if (!PathExists(outputDir)) usage(description);
  for (std::string const& path : graphsPaths)
  {
      if (!PathExists(path)) usage(description);
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
// clang-format on
