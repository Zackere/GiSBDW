// Copyright 2020 GISBDW. All rights reserved.
#pragma once
#include <iostream>
#include <memory>

#include "union_find.hpp"

namespace td {
class UnionFindImpl : public UnionFind {
 public:
  explicit UnionFindImpl(ElemType numberOfElements);
  UnionFindImpl(UnionFindImpl const& uf);
  SetId Union(SetId set1, SetId set2) override;
  SetId Find(ElemType elem) override;
  std::unique_ptr<UnionFind> Clone() override;
  ElemType GetNumberOfElements() override;
  ~UnionFindImpl() override = default;
  ElemType GetMaxValue() override;
  ElemType GetValue(SetId setId) override;

 protected:
  void SetValue(SetId setId, ElemType value) override;

 private:
  ElemType maxValue;
  ElemType numberOfElements;
  std::unique_ptr<ElemType[]> arr;
};
}  // namespace td
