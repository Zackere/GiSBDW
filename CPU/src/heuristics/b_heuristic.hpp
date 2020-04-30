// Copyright 2020 GISBDW. All rights reserved.
#include "../branch_and_bound/branch_and_bound.hpp"
#include "../elimination_tree/elimination_tree.hpp"

namespace td {

class BHeuristic : BranchAndBound::Heuristic {
 public:
  Result Get(BranchAndBound::Graph const& g) override {
    EliminationTree tree(g);

    while (tree.ComponentsBegin() != tree.ComponentsEnd()) {
      auto const& g = tree.ComponentsBegin()->AdjacencyList();
      auto g_iter = g.begin();
      int v = g_iter->first;
      int best_degree = g_iter->second.size();

      while (++g_iter != g.end()) {
        if (g_iter->second.size() > best_degree) {
          v = g_iter->first;
          best_degree = g_iter->second.size();
        }
      }

      tree.Eliminate(v);
    }

    Result result;
    auto [decomp, depth, root] = tree.Decompose();

    result.td_decomp = std::move(decomp);
    result.depth = depth;
    result.root = root;

    return result;
  }
};  // namespace td
}  // namespace td
   // namespace td
