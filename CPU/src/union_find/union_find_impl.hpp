// Copyright 2020 GISBDW. All rights reserved.
#pragma once
#include <iostream>
#include <memory>
#include <vector>
#include "union_find.hpp"

namespace td {
    //Implementation of UnionFind abstract data type in array
class UnionFindImpl : public UnionFind {
 public:
  //Constructor of UnionFindImpl
  //Params
  //numberOfElements - number of starting distinct sets
  explicit UnionFindImpl(ElemType numberOfElements);
  UnionFindImpl(UnionFindImpl const& uf);
  ~UnionFindImpl() override = default;
  SetId Union(SetId set1, SetId set2) override;
  SetId Find(ElemType elem) override;
  std::unique_ptr<UnionFind> Clone() override;
  ElemType GetNumberOfElements() override;
  ElemType GetMaxValue() override;
  ElemType GetValue(SetId setId) override;

 protected:
  void SetValue(SetId setId, ElemType value) override;

 private:
  ElemType maxValue;
  std::vector<ElemType> parents;
};
}  // namespace td
