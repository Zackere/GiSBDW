// Copyright 2020 GISBDW. All rights reserved.
#pragma once
#include <memory>

namespace td {

class UnionFind {
 public:
  using ElemType = int;
  using SetId = int;
  virtual ~UnionFind() = default;
  virtual SetId Find(ElemType elem) = 0;
  virtual SetId Union(SetId set1, SetId set2) = 0;
  virtual std::unique_ptr<UnionFind> Clone() = 0;
  virtual ElemType GetNumberOfElements() = 0;
  virtual ElemType GetMaxValue() =  0;
  virtual ElemType GetValue(SetId setId) = 0;

 protected:
  virtual void SetValue(SetId setId, ElemType value) = 0;
};
}  // namespace td
