// Copyright 2020 GISBDW. All rights reserved.
#pragma once
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "../binomial_coefficients/binomial_coefficient.hpp"
#include "../set_encoder/set_encoder.hpp"
//#include "../quasi_set/quasi_set_array.hpp"
#include "../union_find/array_union_find.hpp"
#include "boost/graph/adjacency_list.hpp"
#include "boost/graph/undirected_graph.hpp"
namespace td {

template <class SignedIntegral>
class DynamicAlgorithm {
 public:
  using Graph =
      boost::adjacency_list<boost::mapS, boost::vecS, boost::undirectedS>;
  // typedef boost::property<boost::vertex_name_t,
  //                        std::string,
  //                        boost::property<boost::vertex_color_t, float> >
  //    vertex_p;
  // using Graph = boost::
  //    adjacency_list<boost::mapS, boost::vecS, boost::undirectedS, vertex_p>;
  using SetElement = std::make_unsigned_t<SignedIntegral>;
  using UnionFind = ArrayUnionFind<SignedIntegral>;

  DynamicAlgorithm() = default;
  ~DynamicAlgorithm() = default;

  SignedIntegral Run(Graph const& graph) {
    if (graph.m_vertices.size() > std::numeric_limits<SetElement>::max()) {
      throw std::out_of_range(
          "Graph has " + std::to_string(graph.m_vertices.size()) +
          ", but specified unsigned version of template argument can hold "
          "maximally up to " +
          std::to_string(std::numeric_limits<SetElement>::max()) + " vertices");
    }
    SetElement n = static_cast<SetElement>(graph.m_vertices.size());
    // return immediately from trivial cases
    if (n == 0)
      return 0;
    if (n == 1)
      return 1;

    // prepare data structures for algorithm
    auto widestPart = NChooseK(n, n / 2);
    std::vector<UnionFind> prevVec;
    std::vector<UnionFind> currVec;
    prevVec.reserve(widestPart);
    currVec.reserve(widestPart);
    prevVec.emplace_back(n);

    // QuasiSetArray<SetElement> quasiSet(n);
    // QuasiSetBase<SetElement>* set = &quasiSet;

    // iterate over k-subsets
    for (SetElement k = 1; k <= n; ++k) {
      auto numberOfSubsets = NChooseK(n, k);
      currVec.clear();
      for (size_t code = 0; code < numberOfSubsets; ++code) {
        currVec.emplace_back(n);
        SetElement bestTreeDepthForThisSet =
            std::numeric_limits<SignedIntegral>::max();
        // get set from its code
        auto set = set_encoder::Decode<std::vector<SetElement>>(n, k, code);
        // set->Decode(code, k);
        for (SetElement elementIndex = 0; elementIndex < set.size();
             ++elementIndex) {
          // exclude one element from set
          // SetElement excludedElement = set->GetElementAtIndex(elementIndex);
          // set->ExcludeTemporarilyElementAtIndex(elementIndex);
          // get a code for a set without this element (it is an index to array
          // containing results from level -1)
          size_t indexToPrevArray =
              set_encoder::Encode(set.data(), set.size(), elementIndex);
          // for convenience bind previous UnionFind structure to variable
          UnionFind& ufPrev = prevVec[indexToPrevArray];
          if (ufPrev.GetMaxValue() < bestTreeDepthForThisSet) {
            UnionFind ufNew(ufPrev);
            for (SetElement index = 0; index < set.size(); ++index) {
              auto elementToCheck = set[index];
              bool areNeighbours =
                  boost::edge(set[elementIndex], elementToCheck, graph).second;
              if (areNeighbours) {
                // if they are neighbours - union sets that represent them
                ufNew.Union(set[elementIndex], ufNew.Find(elementToCheck));
              }
            }
            if (bestTreeDepthForThisSet > ufNew.GetMaxValue()) {
              bestTreeDepthForThisSet = ufNew.GetMaxValue();
              currVec[code] = std::move(ufNew);
            }
          }
          // set->RecoverExcludedElement();
        }
      }
      std::swap(prevVec, currVec);
    }
    SignedIntegral resultTreeDepth = prevVec[0].GetMaxValue();
    return resultTreeDepth;
  }
};
}  // namespace td
