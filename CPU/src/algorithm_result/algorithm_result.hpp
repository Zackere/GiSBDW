#pragma once
#include <filesystem>
#include <string>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
namespace td {
class AlgorithmResult {
 public:
  AlgorithmResult() = default;

  void WriteToFile(std::filesystem::path const& path) {
    std::ofstream file(path);
    boost::property_tree::ptree root;
    root.put("timeElapsed", timeElapsed);
    root.put("treedepth", treedepth);
    boost::property_tree::write_json(file, root);
    file.close();
  }

  int treedepth = -1;
  double timeElapsed = -1;
};

}  // namespace td
