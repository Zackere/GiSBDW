// Copyright 2020 GISBDW. All rights reserved.
#pragma once
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "src/elimination_tree/elimination_tree.hpp"

namespace td {
struct Statistics {
  EliminationTree::Result decomposition;
  double time_elapsed;
};

std::ostream& operator<<(std::ostream& out, Statistics const& s) {
  boost::property_tree::ptree root;
  root.put("timeElapsed", s.time_elapsed);
  root.put("treedepth", s.decomposition.treedepth);
  root.put("root", s.decomposition.root);
  boost::property_tree::write_json(out, root);
  return out;
}
}  // namespace td
