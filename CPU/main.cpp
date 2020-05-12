// Copyright 2020 GISBDW. All rights reserved.
// clang-format off

#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>

#include "src/dynamic_algorithm/dynamic_algorithm.hpp"
#include "src/algorithm_result/algorithm_result.hpp"
#include "boost/graph/adjacency_list.hpp"
#include "boost/graph/erdos_renyi_generator.hpp"
#include "boost/graph/graphviz.hpp"
#include "boost/program_options.hpp"
#include "boost/graph/properties.hpp"

#include "boost/property_map/property_map.hpp"

//Do podmiany (bledy z linkowaniem byly)
//#include <libs\graph\src\read_graphviz_new.cpp>
using namespace boost;
namespace po = boost::program_options;
namespace fs = std::filesystem;

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
    //using Graph =
        //boost::adjacency_list<boost::mapS, boost::vecS, boost::undirectedS>;
    typedef property < vertex_name_t, std::string,
        property < vertex_color_t, float > > vertex_p;
    using Graph =  adjacency_list < mapS, vecS, undirectedS, vertex_p>;
    using ERGen = boost::sorted_erdos_renyi_iterator<std::minstd_rand, Graph>;
    constexpr int n = 16;
    std::minstd_rand rng(0);
    Graph g(ERGen(rng, n, 0.5), ERGen(), n);

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
        if (vm.count("help")) {
            usage(description);
        }
        po::notify(vm);
    }
    catch (po::error & ex)
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
        ////////
        // Construct an empty graph and prepare the dynamic_property_maps.
        Graph graph(0);
        dynamic_properties dp;

        property_map<Graph, vertex_name_t>::type name =
            get(vertex_name, graph);
        dp.property("node_id", name);

        ////////
        std::ifstream graphFile(path);
        bool result = read_graphviz(graphFile, graph, dp, "node_id");
        graphFile.close();

        td::AlgorithmResult algorithmResult; // = execute algorithm(g);


        auto t1 = std::chrono::high_resolution_clock::now();
        if (algorithmType == "bnb")
        {

        }
        else if (algorithmType == "dyn")
        {
            td::DynamicAlgorithm<int> dynamicAlgorithm;
            algorithmResult = dynamicAlgorithm.Run(graph);
        }
        else if (algorithmType == "hyb")
        {

        }
        else
        {
            std::cerr << "Wrong algorithm option specified.\n";
            usage(description);
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() / 1000.0;
        std::cout << "Elapsed -> " << duration << "s\n";
        std::cout << "Is undirected? -> " << boost::is_undirected(g) << "\n";
        std::cout << "Tree depth -> " << algorithmResult.treedepth << "\n";
        fs::path outputFilePath = ((outputPath / path.filename()) += ".out");
        std::cout << "outputFilePath -> " << outputFilePath << "\n";
        algorithmResult.WriteToFile(outputFilePath);
        return 0;
    }
}

  //using ERGen = boost::erdos_renyi_iterator<std::minstd_rand, Graph>;
  //int n = 25;
  //std::minstd_rand rng;
  //Graph g(ERGen(rng, n, 0.05), ERGen(), n);
  //boost::depth_first_search(g, boost::visitor(VertexVisitor()));
  //std::ofstream file("graph.gviz", std::ios_base::trunc);
  //boost::write_graphviz(file, g);
  //file.close();
// clang-format on
