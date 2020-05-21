// Copyright 2020 GISBDW. All rights reserved.
#pragma once
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <string>

#include "src/elimination_tree/elimination_tree.hpp"

namespace td {
struct Statistics {
  std::string graph_name;
  EliminationTree::Result decomposition;
  double time_elapsed;
  std::size_t nedges;
  std::size_t nvertices;
};

std::ostream& operator<<(std::ostream& out, Statistics const& s) {
  boost::property_tree::ptree root;
  root.put("graphName", s.graph_name);
  root.put("timeElapsed", s.time_elapsed);
  root.put("treedepth", s.decomposition.treedepth);
  root.put("root", s.decomposition.root);
  root.put("edges", s.nedges);
  root.put("vertices", s.nvertices);
  boost::property_tree::write_json(out, root);
  return out;
}
}  // namespace td
