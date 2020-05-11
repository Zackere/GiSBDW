#pragma once
#include <filesystem>
#include <string>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#define STRINGIFY(Variable) (#Variable)
namespace td {
struct AlgorithmResult {
 public:
  double timeElapsed = -1;
  int treedepth = -1;

  void WriteToFile(std::filesystem::path const& path) {
    std::ofstream file(path);
    boost::property_tree::ptree root;
    root.put(STRINGIFY(timeElapsed), timeElapsed);
    root.put(STRINGIFY(treedepth), treedepth);
    boost::property_tree::write_json(file, root);
    file.close();
  }
};

}  // namespace td
