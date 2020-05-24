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
#include "src/branch_and_bound/heuristics/bottom_up_heuristic.hpp"
#include "src/branch_and_bound/heuristics/highest_degree_heuristic.hpp"
#include "src/branch_and_bound/heuristics/spanning_tree_heuristic.hpp"
#include "src/branch_and_bound/heuristics/variance_heuristic.hpp"
#include "src/branch_and_bound/lower_bound/basic_lower_bound.hpp"
#include "src/branch_and_bound/lower_bound/edge_lower_bound.hpp"
#include "src/dynamic_cpu/dynamic_cpu.hpp"
#include "src/dynamic_cpu/dynamic_cpu_improv.hpp"
#include "src/elimination_tree/elimination_tree.hpp"
#include "src/statistics/statistics.hpp"

namespace {
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

std::string GetRandomFilenameSuffix() {
  auto epoch = std::chrono::duration_cast<std::chrono::seconds>(
                   std::chrono::high_resolution_clock::now().time_since_epoch())
                   .count();
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<> dist;
  return std::to_string(epoch) + std::to_string(dist(rng));
}
}  // namespace

int main(int argc, char** argv) {
  using Graph =
      boost::adjacency_list<boost::mapS, boost::vecS, boost::undirectedS>;

  std::string algorithm_type;
  std::string output_dir_str;
  std::vector<std::string> graph_paths_string;
  std::vector<fs::path> graph_paths;
  fs::path output_dir;
  po::options_description description("Usage");
  description.add_options()("help", "print this message")(
      "algorithm,a", po::value<std::string>(&algorithm_type)->required(),
      "Select algorithm to run.\n"
      "Possible args:\n"
      "bnbCPU - for branch and bound algorithm ran on CPU\n"
      "dynCPU - for dynamic algorithm ran on CPU\n"
      "dynCPUImprov - for dynamic algorithm ran on CPU version 2\n"
      "highestDegreeHeur - for highest degree heuristic\n"
      "spanningTreeHeur - for spanning tree heuristic\n"
      "varianceHeur - for variance heuristic\n"
      "bottomUpHeur - for union-find based heurstic\n")(
      "input,i",
      po::value<std::vector<std::string>>(&graph_paths_string)->required(),
      "path to input graph")(
      "output,o", po::value<std::string>(&output_dir_str)->required(),
      "path to output dir");

  po::positional_options_description positional_args;
  positional_args.add("input", -1);
  po::variables_map vm;
  try {
    po::store(po::command_line_parser(argc, argv)
                  .options(description)
                  .positional(positional_args)
                  .run(),
              vm);
    if (vm.count("help"))
      Usage(description);
    po::notify(vm);
  } catch (po::error& ex) {
    std::cerr << ex.what() << "\n";
    Usage(description);
  }
  output_dir = fs::path(output_dir_str);
  for (auto const& pathString : graph_paths_string)
    graph_paths.push_back(fs::path(pathString));
  if (!PathExists(output_dir) || !IsDirectory(output_dir))
    Usage(description);
  for (fs::path const& path : graph_paths)
    if (!PathExists(path) || !IsFile(path))
      Usage(description);

  for (fs::path const& graph_path : graph_paths) {
    std::ifstream graph_ifstream(graph_path);
    std::string graph_string{std::istreambuf_iterator<char>(graph_ifstream),
                             std::istreambuf_iterator<char>()};
    graph_ifstream.close();
    std::regex vertex_matcher("^\\d+\\s*;$");
    auto vertices = std::sregex_iterator(
        std::begin(graph_string), std::end(graph_string), vertex_matcher);
    Graph graph(std::distance(vertices, std::sregex_iterator()));
    std::regex edge_matcher("(\\d+)\\s*--\\s*(\\d+)");
    for (auto edges = std::sregex_iterator(
             std::begin(graph_string), std::end(graph_string), edge_matcher);
         edges != std::sregex_iterator(); ++edges)
      boost::add_edge(std::stoi((*edges)[1]), std::stoi((*edges)[2]), graph);

    td::Statistics stats;
    auto graph_name = graph_path;
    while (graph_name != graph_name.stem())
      graph_name = graph_name.stem();
    stats.graph_name = graph_name.string();
    stats.nvertices = boost::num_vertices(graph);
    stats.nedges = boost::num_edges(graph);
    stats.algorithm_type = algorithm_type;
    std::cout << "Processing graph " << stats.graph_name << std::endl;
    std::cout << "Algorithm: " << stats.algorithm_type << std::endl;
    std::cout << "Vertices: " << stats.nvertices << std::endl;
    std::cout << "Edges: " << stats.nedges << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();

    if (algorithm_type == "dynCPU") {
      td::DynamicCPU dcpu;
      dcpu(graph);
      stats.decomposition = dcpu.GetTDDecomp(0, graph);
    } else if (algorithm_type == "dynCPUImprov") {
      td::DynamicCPUImprov dcpu;
      dcpu(graph);
      td::DynamicCPUImprov::CodeType code = 1;
      code <<= boost::num_vertices(graph);
      stats.decomposition = dcpu.GetTDDecomp(--code, graph);
    } else if (algorithm_type == "bnbCPU") {
      td::BranchAndBound bnb;
      stats.decomposition =
          bnb(graph, std::make_unique<td::EdgeLowerBound>(),
              std::make_unique<td::HighestDegreeHeuristic>(
                  std::make_unique<td::SpanningTreeHeuristic>(
                      std::make_unique<td::VarianceHeuristic>(
                          std::make_unique<td::BottomUpHeuristicGPU>(nullptr),
                          1.0, 0.2, 0.8))));
    } else if (algorithm_type == "highestDegreeHeur") {
      stats.decomposition = td::HighestDegreeHeuristic(nullptr).Get(graph);
    } else if (algorithm_type == "spanningTreeHeur") {
      stats.decomposition = td::SpanningTreeHeuristic(nullptr).Get(graph);
    } else if (algorithm_type == "varianceHeur") {
      stats.decomposition =
          td::VarianceHeuristic(nullptr, 1.0, 0.2, 0.8).Get(graph);
    } else if (algorithm_type == "bottomUpHeur") {
      stats.decomposition = td::BottomUpHeuristicGPU(nullptr).Get(graph);
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
    fs::path stats_path = output_dir / graph_name;
    stats_path += "_";
    stats_path += GetRandomFilenameSuffix();
    stats_path += ".out";
    std::cout << "Output written to: " << stats_path << "\n\n";
    std::ofstream file(stats_path);
    file << stats;
    file.close();
    fs::path td_decomp_path = stats_path;
    td_decomp_path += ".gviz";
    file = std::ofstream(td_decomp_path);
    boost::write_graphviz(file, stats.decomposition.td_decomp);
    file.close();
  }

  return 0;
}
