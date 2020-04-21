// Copyright 2020 GISBDW. All rights reserved.
#pragma once
#include <memory>

namespace td {

class UnionFind {
 public:
  using ElemType = int;
  using SetId = int;
  virtual ~UnionFind() = default;
  //Find a representant of set which contains given element.
  //Params
  //elem - element of set which we want to know representant of.
  //Return
  //SetId of set that contains elem
  virtual SetId Find(ElemType elem) = 0;
  //Make a union of two sets.
  //Params
  //set1, set2 - representants of two distinct sets to join
  //Return
  //SetId representant of set1
  virtual SetId Union(SetId set1, SetId set2) = 0;
  //Make a deep copy of UnionFind
  //Return
  //std::unique_ptr<UnionFind> to created copy
  virtual std::unique_ptr<UnionFind> Clone() = 0;
  //Get total number of all elements
  //Return
  //ElemType total number of all elements
  virtual ElemType GetNumberOfElements() = 0;
  //Get the largest currently value associated with any set
  //Return
  //ElemType the largest currently associated value with any set
  virtual ElemType GetMaxValue() =  0;
  //Get value associated with given set
  //Params
  //setId - setId of set for which we want to get value associated value
  //Return
  //ElemType value associated with set represented by given setId
  virtual ElemType GetValue(SetId setId) = 0;
};
}  // namespace td
