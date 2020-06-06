// Copyright 2020 GISBDW. All rights reserved.
#include "dynamic_cpu.hpp"

#include <algorithm>
#include <boost/graph/iteration_macros.hpp>
#include <limits>
#include <set>

#include "src/set_encoder/set_encoder.hpp"

namespace td {
namespace {
template <typename Callback>
void SearchGraphOnVertices(DynamicCPU::BoostGraph const& g,
                           std::set<int> const& verts,
                           Callback callback,
                           int currentVertex) {
  if (verts.find(currentVertex) == std::end(verts) || callback(currentVertex))
    return;

  BGL_FORALL_ADJ_T(currentVertex, neigh, g, DynamicCPU::BoostGraph) {
    SearchGraphOnVertices(g, verts, callback, neigh);
  }
}
}  // namespace
std::size_t DynamicCPU::GetIterationsPerformed() const {
  return history_.size();
}

std::size_t DynamicCPU::GetTreedepth(std::size_t nverts,
                                     std::size_t subset_size,
                                     std::size_t subset_code) const {
  if (subset_size < history_.size()) {
    if (auto it = history_[subset_size].find(subset_code);
        it != std::end(history_[subset_size]))
      return std::get<0>(it->second);
  }
  return std::numeric_limits<std::size_t>::max();
}

void DynamicCPU::Run(BoostGraph const& g) {
  history_.clear();
  history_.reserve(0);
  history_.reserve(boost::num_vertices(g) + 1);
  history_.emplace_back();
  history_[0].insert({0, {0, -1}});
  for (std::size_t i = 1; i < history_.capacity(); ++i) {
    history_.emplace_back();
    auto nk = set_encoder::NChooseK(boost::num_vertices(g), i);
    for (std::size_t code = 0; code < nk; ++code) {
      auto set =
          set_encoder::Decode<std::set<int>>(boost::num_vertices(g), i, code);
      std::set<int> visited;
      SearchGraphOnVertices(g, set,
                            [&visited](auto v) {
                              if (visited.find(v) != std::end(visited))
                                return true;
                              visited.insert(v);
                              return false;
                            },
                            *std::begin(set));
      if (visited.size() != set.size())
        continue;

      std::size_t proccesed = 0;
      std::size_t treedepth = std::numeric_limits<std::size_t>::max();
      int root;
      for (auto it = std::begin(set); it != std::end(set); ++it) {
        auto v = *it;
        it = set.erase(it);

        std::size_t component_treedepth = 0;
        std::vector<bool> visited_total(boost::num_vertices(g));
        visited_total[v] = true;

        BGL_FORALL_ADJ_T(v, neigh, g, DynamicCPU::BoostGraph) {
          if (!visited_total[neigh]) {
            visited.clear();
            SearchGraphOnVertices(g, set,
                                  [&visited, &visited_total](auto v) {
                                    if (visited.find(v) != std::end(visited))
                                      return true;
                                    visited.insert(v);
                                    visited_total[v] = true;
                                    return false;
                                  },
                                  neigh);
            component_treedepth = std::max(
                component_treedepth,
                std::get<0>(
                    history_[visited.size()][set_encoder::Encode(visited)]));
          }
        }
        if (component_treedepth + 1 < treedepth) {
          treedepth = component_treedepth + 1;
          root = v;
        }

        it = set.insert(it, v);
        ++proccesed;
      }
      history_[i][code] = {treedepth, root};
    }
  }
}
}  // namespace td
