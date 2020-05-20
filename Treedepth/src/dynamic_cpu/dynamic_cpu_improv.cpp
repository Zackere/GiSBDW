#include "dynamic_cpu_improv.hpp"

#include <tuple>

namespace td {
namespace {
template <typename Key, typename T, typename Allocator>
std::size_t Encode(std::map<Key, T, std::less<Key>, Allocator> const& m) {
  std::size_t ret = 0;
  for (auto& p : m)
    ret |= static_cast<std::size_t>(1) << p.first;
  return ret;
}
}  // namespace
std::size_t DynamicCPUImprov::Run(
    EliminationTree::ComponentIterator component) {
  auto code = Encode(component->AdjacencyList());
  auto tdinfoit = history_[component->AdjacencyList().size()].find(code);
  if (tdinfoit == std::end(history_[component->AdjacencyList().size()])) {
    auto& p = history_[component->AdjacencyList().size()][code];
    p = {std::numeric_limits<std::size_t>::max(), -1};
    for (auto it = std::begin(component->AdjacencyList());
         it != std::end(component->AdjacencyList()); ++it) {
      auto v = it->first;
      auto comps = et_->Eliminate(v);

      std::size_t tdcomp = 0;
      for (auto c : comps)
        tdcomp = std::max(tdcomp, Run(c));
      ++tdcomp;

      if (std::get<0>(p) > tdcomp)
        p = {tdcomp, v};

      std::tie(component, it) = et_->Merge();
    }
    return std::get<0>(p);
  }
  return std::get<0>(tdinfoit->second);
}

EliminationTree::Result DynamicCPUImprov::GetTDDecompImpl(std::size_t code,
                                                          BoostGraph const& g) {
  if (boost::num_vertices(g) < history_.size())
    if (auto it = history_[boost::num_vertices(g)].find(code);
        it != std::end(history_[boost::num_vertices(g)])) {
      td::EliminationTree et(g);
      while (et.ComponentsBegin() != et.ComponentsEnd())
        et.Eliminate(std::get<1>(
            history_[et.ComponentsBegin()->AdjacencyList().size()]
                    [Encode(et.ComponentsBegin()->AdjacencyList())]));
      return et.Decompose();
    }
  return EliminationTree::Result{EliminationTree::BoostGraph(),
                                 std::numeric_limits<unsigned>::max(), 0};
}
}  // namespace td
