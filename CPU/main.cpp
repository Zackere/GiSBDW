// Copyright 2020 GISBDW. All rights reserved.
// clang-format off

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <filesystem>

#include "src/algorithm_result/algorithm_result.hpp"
#include "boost/graph/adjacency_list.hpp"
#include "boost/graph/depth_first_search.hpp"
#include "boost/graph/erdos_renyi_generator.hpp"
#include "boost/graph/graphviz.hpp"
#include "boost/program_options.hpp"
//Do podmiany (bledy z linkowaniem byly)
//#include <libs\graph\src\read_graphviz_new.cpp>

namespace po = boost::program_options;
namespace fs = std::filesystem;
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

bool PathExists(fs::path const & path)
{
    if (!fs::exists(path))
    {
        std::cerr << path << " does not exist.\n";
        return false;
    }
    return true;
}
bool IsDirectory(fs::path const& path)
{
    if (!fs::is_directory(path))
    {
        std::cerr << path << " is not a directory.\n";
        return false;
    }
    return true;
}

bool IsFile(fs::path const& path)
{
    if (!fs::is_regular_file(path))
    {
        std::cerr << path << " is not a regular file.\n";
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
  using Graph = boost::adjacency_list<>;

    std::string algorithmType;
    std::string outputDirString;
    std::vector<std::string> graphsPathsStrings;
    std::vector<fs::path> graphPaths;
    fs::path outputPath;
  po::options_description description("Usage");
  description.add_options()
      ("help", "print this message")(
      "algorithm,a", po::value<std::string>(&algorithmType)->required(),
          "Select algorithm to run.\n"
          "Possible args:\n"
          "bnb - for branch and bound algorithm\n"
          "dyn - for dynamic algorithm\n"
          "hyb - for hybrid algorithm\n")(
      "input,i", po::value<std::vector<std::string>>(&graphsPathsStrings)->required(), "path to input graph")(
      "output,o", po::value<std::string>(&outputDirString)->required(), "path to output dir");

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
  outputPath = fs::path(outputDirString);
  for (auto const& pathString : graphsPathsStrings)
  {
      graphPaths.push_back(fs::path(pathString));
  }
  if (!PathExists(outputPath) || !IsDirectory(outputPath)) usage(description);
  for (fs::path const& path : graphPaths)
  {
      if (!PathExists(path) || !IsFile(path)) usage(description);
  }

  for (fs::path const& path : graphPaths)
  {
      Graph g;
      boost::dynamic_properties dp;
      //std::ifstream graphFile(path);
      //bool result = boost::read_graphviz(graphFile, g, dp);
      //graphFile.close();

      td::AlgorithmResult algorithmResult; // = execute algorithm(g);
      //algorithmResult.WriteToFile(std::filesystem::path::append()
      fs::path outputFilePath = ((outputPath / path.filename()) += ".out");
      std::cout << "outputFilePath -> " << outputFilePath << "\n";
      //algorithmResult.WriteToFile(outputFilePath);

  }


  //using ERGen = boost::erdos_renyi_iterator<std::minstd_rand, Graph>;
  //int n = 25;
  //std::minstd_rand rng;
  //Graph g(ERGen(rng, n, 0.05), ERGen(), n);
  //boost::depth_first_search(g, boost::visitor(VertexVisitor()));
  //std::ofstream file("graph.gviz", std::ios_base::trunc);
  //boost::write_graphviz(file, g);
  //file.close();
  return 0;
}
// clang-format on
