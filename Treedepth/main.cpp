// Copyright 2020 GISBDW. All rights reserved.

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "boost/graph/adjacency_list.hpp"
#include "boost/graph/erdos_renyi_generator.hpp"
#include "boost/graph/graphviz.hpp"

#include "boost/graph/properties.hpp"
#include "boost/program_options.hpp"
#include "src/algorithm_result/algorithm_result.hpp"
#include "src/dynamic_algorithm/dynamic_algorithm.hpp"

#include "boost/property_map/property_map.hpp"

namespace po = boost::program_options;
namespace fs = std::filesystem;
namespace bo = boost;
void usage(po::options_description const& description) {
  std::cout << description;
  std::cout << "Example usage: ./app.exe -a bnb -o /path/to/output/dir "
               "/path/to/graph1 /path/to/graph2\n";
  std::exit(1);
}

bool PathExists(fs::path const& path) {
  if (!fs::exists(path)) {
    std::cerr << path << " does not exist.\n";
    return false;
  }
  return true;
}
bool IsDirectory(fs::path const& path) {
  if (!fs::is_directory(path)) {
    std::cerr << path << " is not a directory.\n";
    return false;
  }
  return true;
}

bool IsFile(fs::path const& path) {
  if (!fs::is_regular_file(path)) {
    std::cerr << path << " is not a regular file.\n";
    return false;
  }
  return true;
}

int main(int argc, char** argv) {
  // using Graph =
  // boost::adjacency_list<boost::mapS, boost::vecS, boost::undirectedS>;
  typedef bo::property<bo::vertex_name_t, std::string,
                       bo::property<bo::vertex_color_t, float>>
      vertex_p;
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
