// Copyright 2020 GISBDW. All rights reserved.

#include <limits>
#include <memory>

#include "src/branch_and_bound/branch_and_bound.hpp"
#include "src/branch_and_bound/heuristics/bottom_up_heuristic_gpu_algorithm.hpp"

namespace td {
class BottomUpHeuristicGPU : public BranchAndBound::Heuristic {
 public:
  explicit BottomUpHeuristicGPU(
      std::unique_ptr<BranchAndBound::Heuristic> heuristic)
      : heuristic_(std::move(heuristic)) {}
  ~BottomUpHeuristicGPU() override = default;

  EliminationTree::Result Get(BranchAndBound::Graph const& g) override {
    EliminationTree result(g);
    BottomUpHeuristicGPUAlgorithm dgpu;
    int n = boost::num_vertices(g);
    dgpu(g);

    if (dgpu.GetIterationsPerformed() == n + 1) {
      for (auto v : dgpu.GetElimination<EliminationTree::VertexType>(n, n, 0))
        result.Eliminate(v);

      return result.Decompose();
    }

    return EliminationTree::Result{EliminationTree::BoostGraph(),
                                   std::numeric_limits<unsigned>::max(), 0};
  }

 private:
  std::unique_ptr<BranchAndBound::Heuristic> heuristic_;
};
}  // namespace td
