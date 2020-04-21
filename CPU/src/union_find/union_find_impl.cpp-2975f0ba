// Copyright 2020 GISBDW. All rights reserved.
#include "union_find_impl.hpp"
#include <algorithm>
namespace td {

UnionFindImpl::UnionFindImpl(ElemType numberOfElements)
    : numberOfElements(numberOfElements),
      maxValue(-1),
      parents(new ElemType[numberOfElements]) {
  std::fill(parents.get(), parents.get() + numberOfElements, -1);
}

UnionFindImpl::UnionFindImpl(UnionFindImpl const& uf)
    : numberOfElements(uf.numberOfElements),
      maxValue(uf.maxValue),
      parents(new ElemType[numberOfElements]) {
  std::copy(uf.parents.get(), uf.parents.get() + numberOfElements,
            parents.get());
}

UnionFind::SetId UnionFindImpl::Find(ElemType elem) {
  ElemType iterator = elem;
  while (parents[iterator] >= 0) {
    iterator = parents[iterator];
  }
  // path compression
  ElemType tmp;
  ElemType compressionIterator = elem;
  while (compressionIterator != iterator) {
    tmp = parents[compressionIterator];
    parents[compressionIterator] = iterator;
    compressionIterator = tmp;
  }
  return SetId(iterator);
}
UnionFind::SetId UnionFindImpl::Union(SetId set1, SetId set2) {
  ElemType set1Val = GetValue(set1);
  ElemType set2Val = GetValue(set2);
  parents[set2] = set1;
  SetValue(set1, set1Val > set2Val ? set1Val : set2Val + 1);
  return set1;
}
std::unique_ptr<UnionFind> UnionFindImpl::Clone() {
  return std::make_unique<UnionFindImpl>(*this);
}

UnionFind::ElemType UnionFindImpl::GetNumberOfElements() {
  return numberOfElements;
}
UnionFind::ElemType UnionFindImpl::GetMaxValue() {
  return maxValue;
}
UnionFind::ElemType UnionFindImpl::GetValue(SetId setId) {
  return -parents[setId];
}
void UnionFindImpl::SetValue(SetId setId, ElemType value) {
  if (value > maxValue)
    maxValue = value;
  parents[setId] = -value;
}
}  // namespace td
