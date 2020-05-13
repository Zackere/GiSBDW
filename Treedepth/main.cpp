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
void usage(po::options_description const &description) {
  std::cout << description;
  std::cout << "Example usage: ./app.exe -a bnb -o /path/to/output/dir "
               "/path/to/graph1 /path/to/graph2\n";
  std::exit(1);
}

bool PathExists(fs::path const &path) {
  if (!fs::exists(path)) {
    std::cerr << path << " does not exist.\n";
    return false;
  }
  return true;
}
bool IsDirectory(fs::path const &path) {
  if (!fs::is_directory(path)) {
    std::cerr << path << " is not a directory.\n";
    return false;
  }
  return true;
}

bool IsFile(fs::path const &path) {
  if (!fs::is_regular_file(path)) {
    std::cerr << path << " is not a regular file.\n";
    return false;
  }
  return true;
}

int main(int argc, char **argv) {
  // using Graph =
  // boost::adjacency_list<boost::mapS, boost::vecS, boost::undirectedS>;
  typedef bo::property<bo::vertex_name_t, std::string,
                       bo::property<bo::vertex_color_t, float>>
      vertex_p;
  using Graph =

      bo::adjacency_list<bo::mapS, bo::vecS, bo::undirectedS, vertex_p>;

  td::AlgorithmResult foo;

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
      "bnb - for branch and bound algorithm\n"
      "dyn - for dynamic algorithm\n"
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
    if (vm.count("help")) {
      usage(description);
    }
    po::notify(vm);
  } catch (po::error &ex) {
    std::cerr << ex.what() << "\n";
    usage(description);
  }
  outputPath = fs::path(outputDirString);
  for (auto const &pathString : graphsPathsStrings) {
    graphPaths.push_back(fs::path(pathString));
  }
  if (!PathExists(outputPath) || !IsDirectory(outputPath))
    usage(description);
  for (fs::path const &path : graphPaths) {
    if (!PathExists(path) || !IsFile(path))
      usage(description);
  }

  for (fs::path const &path : graphPaths) {
    Graph graph(0);
    bo::dynamic_properties dp;

    bo::property_map<Graph, bo::vertex_name_t>::type name =
        get(bo::vertex_name, graph);
    dp.property("node_id", name);

    std::ifstream graphFile(path);
    try {
      bool result = read_graphviz(graphFile, graph, dp, "node_id");
    } catch (boost::bad_graphviz_syntax &ex) {
      std::cerr << path << " is not a proper graphviz file. Skipping.\n";
      continue;
    }
    graphFile.close();

    td::AlgorithmResult algorithmResult;
    std::cout << "Processing graph " << path.filename() << std::endl;
    std::cout << "Vertices: " << graph.m_vertices.size() << std::endl;
    std::cout << "Edges: " << graph.m_edges.size() << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();

    if (algorithmType == "dyn") {
      td::DynamicAlgorithm<int> dynamicAlgorithm;
      algorithmResult.treedepth = dynamicAlgorithm.Run(graph);
    } else {
      std::cerr << "Wrong algorithm option specified.\n";
      usage(description);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() /
        1000.0;
    algorithmResult.timeElapsed = duration;
    std::cout << "Elapsed time: " << duration << " seconds\n";
    std::cout << "Treedepth: " << algorithmResult.treedepth << "\n";
    fs::path outputFilePath = ((outputPath / path.filename()) += ".out");
    std::cout << "Output written to: " << outputFilePath << "\n\n";
    algorithmResult.WriteToFile(outputFilePath);
  }

  return 0;
}
