#pragma once
using namespace boost;
//#include <algorithm>  // for std::for_each
//#include <boost/graph/dijkstra_shortest_paths.hpp>
//#include <boost/graph/graph_traits.hpp>
//#include <utility>  // for std::pair
//#include <iostream>  // for std::cout
#include <boost/graph/adjacency_list.hpp>
#include <limits>
#include <memory>
#include <set>
#include <type_traits>
#include <vector>
#include "../binomial_coefficients/binomial_coefficient.hpp"
#include "../set/quasi_set_array.hpp"
#include "../union_find/array_union_find.hpp"
#include "boost/graph/undirected_graph.hpp"

namespace td {

template <class SignedIntegral>
class DynamicAlgorithm {
 public:
  using Graph =
      boost::adjacency_list<boost::mapS, boost::vecS, boost::undirectedS>;

  // using Graph = boost::adjacency_list<>;
  using SetElement = std::make_unsigned_t<SignedIntegral>;
  using UnionFind = ArrayUnionFind<SignedIntegral>;

  DynamicAlgorithm() = default;
  ~DynamicAlgorithm() = default;

  size_t ComputeTreeDepth(Graph const& graph) {
    if (graph.m_vertices.size() > std::numeric_limits<SetElement>::max()) {
      throw std::out_of_range(
          "Graph has " + std::to_string(graph.m_vertices.size()) +
          ", but specified unsigned version of template argument can hold "
          "maximally " +
          std::to_string(std::numeric_limits<SetElement>::max()) + "vertices");
    }
    SetElement n = static_cast<SetElement>(graph.m_vertices.size());
    // return immediately from trivial cases
    if (n == 0)
      return 0;
    if (n == 1)
      return 1;

    // prepare data structures for algorithm
    auto widestPart = NChooseK(n, n / 2);
    std::vector<UnionFind> vec1(widestPart, UnionFind(n));
    std::vector<UnionFind> vec2(widestPart, UnionFind(n));

    std::vector<UnionFind>* prevVec = &vec1;
    std::vector<UnionFind>* currVec = &vec2;

    // SPRAWDZ CAPACITY TYCH VECTOROW GOSCIU!!!!;
    QuasiSetArray<SetElement> quasiSet(n);
    QuasiSetBase<SetElement>* set = &quasiSet;
    // bind indexes of vectors[2] to logical meaning

    // iterate over k-subsets
    for (SetElement k = 2; k <= n; ++k) {
      std::cout << k << "\n";
      auto numberOfSubsets = NChooseK(n, k);
      for (size_t code = 0; code < numberOfSubsets; ++code) {
        // std::cout << "k/code kmax/codemax -> " << k << "/" << code << " " <<
        // n
        //<< "/" << numberOfSubsets << "\n";
        SetElement bestTreeDepthForThisSet =
            std::numeric_limits<SignedIntegral>::max();
        // get set from its code
        set->Decode(code, k);
        for (SetElement elementIndex = 0;
             elementIndex < set->GetNumberOfElements(); ++elementIndex) {
          // exclude one element from set
          SetElement excludedElement = set->GetElementAtIndex(elementIndex);
          set->ExcludeTemporarilyElementAtIndex(elementIndex);
          // get a code for a set without this element (it is an index to array
          // containing results from level -1)
          size_t indexToPrevArray = set->EncodeExcluded();
          // for convenience bind previous UnionFind structure to variable
          UnionFind& ufPrev = (*prevVec)[indexToPrevArray];
          // to
          if (ufPrev.GetMaxValue() < bestTreeDepthForThisSet) {
            UnionFind ufNew(ufPrev);
            for (SetElement index = 0; index < set->GetNumberOfElements();
                 ++index) {
              auto elementToCheck = set->GetElementAtIndex(index);
              auto pair = boost::edge(excludedElement, elementToCheck, graph);
              // if excluded element is incident to set[index] element
              if (pair.second) {
                auto representative = ufNew.Find(elementToCheck);
                if (representative != excludedElement) {
                  ufNew.Union(ufNew.Find(excludedElement),
                              ufNew.Find(representative));
                }
              }
            }
            // tutaj moze byc nie tak. Znowu porownac tree depth?
            bestTreeDepthForThisSet = ufNew.GetMaxValue();
            (*currVec)[code] = std::move(ufNew);
          }
          set->RecoverExcludedElement();
        }
      }
      std::swap(prevVec, currVec);
    }
    return (*prevVec)[0].GetMaxValue();
  }
  void Test() { std::cout << "dupa;"; }

  void ShowUnionFind(UnionFind& uf) {
    std::cout << "MaxVal -> " << uf.GetMaxValue() << "\n";

    for (int i = 0; i < 6; ++i) {
      std::cout << i << "Find -> " << uf.Find(i) << "\n";
    }
    for (int i = 0; i < 6; ++i) {
      std::cout << i << " GetValue -> " << uf.GetValue(i) << "\n";
    }
  }
};
}  // namespace td
// namespace td

// using namespace boost;

// int main(int, char*[]) {
//  // create a typedef for the Graph type
//  typedef adjacency_list<vecS, vecS, bidirectionalS> Graph;
//
//  // Make convenient labels for the vertices
//  enum { A, B, C, D, E, N };
//  const int num_vertices = N;
//  const char* name = "ABCDE";
//
//  // writing out the edges in the graph
//  typedef std::pair<int, int> Edge;
//  Edge edge_array[] = {Edge(A, B), Edge(A, D), Edge(C, A), Edge(D, C),
//                       Edge(C, E), Edge(B, D), Edge(D, E)};
//  const int num_edges = sizeof(edge_array) / sizeof(edge_array[0]);
//
//  // declare a graph object
//  Graph g(num_vertices);
//
//  // add the edges to the graph object
//  for (int i = 0; i < num_edges; ++i)
//    add_edge(edge_array[i].first, edge_array[i].second, g);
//  ... return 0;
//}
