// Copyright 2020 GISBDW. All rights reserved.

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/erdos_renyi_generator.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/properties.hpp>
#include <boost/program_options.hpp>
#include <boost/property_map/property_map.hpp>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <regex>
#include <string>
#include <vector>

#include "src/branch_and_bound/branch_and_bound.hpp"
#include "src/branch_and_bound/heuristics/highest_degree_heuristic.hpp"
#include "src/branch_and_bound/heuristics/spanning_tree_heuristic.hpp"
#include "src/branch_and_bound/heuristics/variance_heuristic.hpp"
#include "src/branch_and_bound/lower_bound/edge_lower_bound.hpp"
#include "src/dynamic_cpu/dynamic_cpu_improv.hpp"
#include "src/dynamic_gpu/dynamic_gpu.hpp"
#include "src/elimination_tree/elimination_tree.hpp"
#include "src/statistics/statistics.hpp"

namespace po = boost::program_options;
namespace fs = std::filesystem;
namespace bo = boost;
void Usage(po::options_description const& description) {
  std::cout << description;
  std::cout << "Example usage: ./app.exe -a bnbCPU -o /path/to/output/dir "
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
  using Graph =
      boost::adjacency_list<boost::mapS, boost::vecS, boost::undirectedS>;

  td::Statistics foo;

  std::string algorithmType;
  std::string outputDirString;
  std::vector<std::string> graphsPathsStrings;
  std::vector<fs::path> graphPaths;
  fs::path outputPath;
  po::options_description description("Usage");
  description.add_options()("help", "print this message")(
      "algorithm,a", po::value<std::string>(&algorithmType)->required(),
      "Select algorithm to run.\n"
      "Possible args:\n"
      "bnbCPU - for branch and bound algorithm ran on CPU\n"
      "dynCPU - for dynamic algorithm ran on CPU\n"
      "dynGPU - for dynamic algorithm ran on GPU\n"
      "hyb - for hybrid algorithm\n")(
      "input,i",
      po::value<std::vector<std::string>>(&graphsPathsStrings)->required(),
      "path to input graph")(
      "output,o", po::value<std::string>(&outputDirString)->required(),
      "path to output dir");

  po::positional_options_description positionalArgs;
  positionalArgs.add("input", -1);
  po::variables_map vm;
  try {
    po::store(po::command_line_parser(argc, argv)
                  .options(description)
                  .positional(positionalArgs)
                  .run(),
              vm);
    if (vm.count("help"))
      Usage(description);
    po::notify(vm);
  } catch (po::error& ex) {
    std::cerr << ex.what() << "\n";
    Usage(description);
  }
  outputPath = fs::path(outputDirString);
  for (auto const& pathString : graphsPathsStrings)
    graphPaths.push_back(fs::path(pathString));
  if (!PathExists(outputPath) || !IsDirectory(outputPath))
    Usage(description);
  for (fs::path const& path : graphPaths)
    if (!PathExists(path) || !IsFile(path))
      Usage(description);

  for (fs::path const& path : graphPaths) {
    std::ifstream file3(path);
    std::string data{std::istreambuf_iterator<char>(file3),
                     std::istreambuf_iterator<char>()};
    file3.close();
    std::regex vertex_matcher("^\\d+\\s*;$");
    auto vertices =
        std::sregex_iterator(std::begin(data), std::end(data), vertex_matcher);
    Graph graph(std::distance(vertices, std::sregex_iterator()));
    std::regex edge_matcher("(\\d+)\\s*--\\s*(\\d+)");
    for (auto edges = std::sregex_iterator(std::begin(data), std::end(data),
                                           edge_matcher);
         edges != std::sregex_iterator(); ++edges)
      boost::add_edge(std::stoi((*edges)[1]), std::stoi((*edges)[2]), graph);

    td::Statistics stats;
    std::cout << "Processing graph " << path.filename() << std::endl;
    std::cout << "Vertices: " << graph.m_vertices.size() << std::endl;
    std::cout << "Edges: " << graph.m_edges.size() << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();

    if (algorithmType == "dynCPU") {
      td::DynamicCPUImprov dcpu;
      dcpu(graph);
      std::size_t code = (1 << boost::num_vertices(graph)) - 1;
      stats.decomposition = dcpu.GetTDDecomp(code, graph);
    } else if (algorithmType == "dynGPU") {
      td::DynamicGPU dgpu;
      dgpu(graph);
      if (dgpu.GetIterationsPerformed() == boost::num_vertices(graph) + 1) {
        td::EliminationTree et(graph);
        for (auto v : dgpu.GetElimination<td::EliminationTree::VertexType>(
                 boost::num_vertices(graph), boost::num_vertices(graph), 0))
          et.Eliminate(v);
        stats.decomposition = et.Decompose();
      }
    } else if (algorithmType == "bnbCPU") {
      td::BranchAndBound bnb;
      auto res = bnb(graph, std::make_unique<td::EdgeLowerBound>(),
                     std::make_unique<td::HighestDegreeHeuristic>(
                         std::make_unique<td::SpanningTreeHeuristic>(
                             std::make_unique<td::VarianceHeuristic>(
                                 nullptr, 1.0, 0.2, 0.8))));
      stats.decomposition = res;
    } else {
      std::cerr << "Wrong algorithm option specified.\n";
      Usage(description);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() /
        1000.0;
    stats.time_elapsed = duration;
    std::cout << "Elapsed time: " << duration << " seconds\n";
    std::cout << "Treedepth: " << stats.decomposition.treedepth << "\n";
    fs::path outputFilePath = ((outputPath / path.filename()) += ".out");
    std::cout << "Output written to: " << outputFilePath << "\n\n";
    std::ofstream file(outputFilePath);
    file << stats;
    file.close();
  }

  return 0;
}
