// Copyright 2020 GISBDW. All rights reserved.
#pragma once

#include "../branch_and_bound/branch_and_bound.hpp"
#include "../dynamic_gpu/dynamic_gpu.hpp"
#include "../elimination_tree/elimination_tree.hpp"
#include "../set_encoder/set_encoder.hpp"

namespace td {
class DynamicGPULowerBound : public BranchAndBound::LowerBound {
 public:
  ~DynamicGPULowerBound() override = default;
  DynamicGPULowerBound(EliminationTree::BoostGraph const& g,
                       std::unique_ptr<BranchAndBound::LowerBound> prev)
      : nverts_(boost::num_vertices(g)), prev_(std::move(prev)) {
    dgpu_(g);
    std::cout << "Dynamic algorithm performed "
              << dgpu_.GetIterationsPerformed() << " iterations\n";
  }
  std::variant<LowerBoundInfo, TreedepthInfo> Get(
      EliminationTree::Component const& g) override {
    if (g.AdjacencyList().size() >= dgpu_.GetIterationsPerformed())
      if (prev_)
        return prev_->Get(g);
      else
        return LowerBoundInfo{1, std::nullopt};
    auto code = set_encoder::Encode(g.AdjacencyList());
    return TreedepthInfo{
        dgpu_.GetTreedepth(nverts_, g.AdjacencyList().size(), code),
        dgpu_.GetElimination<EliminationTree::VertexType>(
            nverts_, g.AdjacencyList().size(), code)};
  }

 private:
  std::size_t nverts_;
  DynamicGPU dgpu_;
  std::unique_ptr<BranchAndBound::LowerBound> prev_;
};
}  // namespace td
