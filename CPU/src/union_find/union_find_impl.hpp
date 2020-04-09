#pragma once
#include <iostream>

#include "union_find.hpp"

namespace td {
class UnionFindImpl : public UnionFind {
 public:
  UnionFindImpl(ElemType numberOfElements);
  UnionFindImpl(UnionFindImpl const& uf);
  virtual SetId Union(SetId set1, SetId set2) override;
  virtual SetId Find(ElemType elem) override;
  virtual std::unique_ptr<UnionFind> Clone() override;
  virtual ElemType GetNumberOfElements() override;
  ~UnionFindImpl() override = default;
  virtual ElemType GetMaxValue() override;
  virtual ElemType GetValue(SetId setId) override;
  virtual void SetValue(SetId setId, ElemType value) override;

 private:
  ElemType maxValue;
  ElemType numberOfElements;
  std::unique_ptr<ElemType[]> arr;
};
}  // namespace td
