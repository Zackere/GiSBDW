// Copyright 2020 GISBDW. All rights reserved.
#include <memory>

#include "union_find_impl.hpp"

namespace td {

UnionFindImpl::UnionFindImpl(ElemType numberOfElements)
    : maxValue(-1),
      numberOfElements(numberOfElements),
      arr(std::make_unique<ElemType[]>(numberOfElements)) {
  for (ElemType i = 0; i < numberOfElements; ++i) {
    arr[i] = -1;
  }
}

UnionFindImpl::UnionFindImpl(UnionFindImpl const& uf)
    : maxValue(uf.maxValue),
      numberOfElements(uf.numberOfElements),
      arr(std::make_unique<ElemType[]>(numberOfElements)) {
  for (ElemType i = 0; i < numberOfElements; ++i) {
    arr[i] = uf.arr[i];
  }
}

UnionFind::SetId UnionFindImpl::Find(ElemType elem) {
  ElemType iterator = elem;
  while (arr[iterator] >= 0) {
    iterator = arr[iterator];
  }
  // path compression
  ElemType tmp;
  ElemType compressionIterator = elem;
  while (compressionIterator != iterator) {
    tmp = arr[compressionIterator];
    arr[compressionIterator] = iterator;
    compressionIterator = tmp;
  }
  return SetId(iterator);
}
UnionFind::SetId UnionFindImpl::Union(SetId set1, SetId set2) {
  ElemType set1Val = GetValue(set1);
  ElemType set2Val = GetValue(set2);
  arr[set2] = set1;
  SetValue(set1, set1Val > set2Val ? set1Val : set2Val + 1);
  return set1;
}
std::unique_ptr<UnionFind> UnionFindImpl::Clone() {
  return std::make_unique<UnionFindImpl>(*this);
}

UnionFind::ElemType UnionFindImpl::GetNumberOfElements() {
  return numberOfElements;
}
UnionFind::ElemType UnionFindImpl::GetMaxValue() { return maxValue; }
UnionFind::ElemType UnionFindImpl::GetValue(SetId setId) { return -arr[setId]; }
void UnionFindImpl::SetValue(SetId setId, ElemType value) {
  if (value > maxValue) maxValue = value;
  arr[setId] = -value;
}
}  // namespace td
