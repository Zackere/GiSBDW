// Copyright 2020 GISBDW. All rights reserved.
#pragma once

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace td {
/**
 * Array UnionFind implementation. It requires template argument to be a numeric
 * signed type e.g. int8_t, int, etc. It satisfies contract defined in
 * introductory implementation.
 */
template <typename T>
class ArrayUnionFind {
 public:
  using SignedIntegral = T;
  using UnsignedIntegral = std::make_unsigned_t<T>;
  using ElemType = UnsignedIntegral;
  using SetIdType = UnsignedIntegral;
  using ValueType = UnsignedIntegral;
  explicit ArrayUnionFind(SignedIntegral nelems);
  ArrayUnionFind(ArrayUnionFind const& other);
  ArrayUnionFind(ArrayUnionFind&&) = default;
  ArrayUnionFind& operator=(ArrayUnionFind&&) = default;

  /**
   * Sums two given sets. As a result of this operation:
   * - s1 will not change its id
   * - every element of s2 will be contained in set1
   * - value associated with s1 is replaced with the greates of values:
   * GetValue(s1), GetValue(s2) + 1.
   *
   * @param s1 lhs of Union.
   * @param s2 rhs of Union.
   *
   * @return Id of a resulting set.
   */
  SetIdType Union(SetIdType s1, SetIdType s2);
  /**
   * Returns id of a set elem is contained in. Performs path compression.
   *
   * @param elem query element.
   *
   * @return Id of a set elem is contained in.
   */
  SetIdType Find(ElemType elem);
  /**
   * Returns id of a set elem is contained in. Does not perform path
   * compression.
   *
   * @param elem query element.
   *
   * @return Id of a set elem is contained in.
   */
  SetIdType Find(ElemType elem) const;
  /**
   * @return Value associated with given set.
   */
  ValueType GetValue(SetIdType set_id) const;
  /**
   * @return The greatest of values associated with stored sets.
   */
  ValueType GetMaxValue() const;

 private:
  void SetValue(SetIdType set_id, ValueType value);

  ValueType max_value_ = 1;
  UnsignedIntegral nelems_ = 0;
  std::unique_ptr<SignedIntegral[]> parents_ = nullptr;
};

template <typename T>
inline ArrayUnionFind<T>::ArrayUnionFind(T nelems)
    : max_value_(1),
      nelems_(std::max(static_cast<T>(0), nelems)),
      parents_(new SignedIntegral[nelems_]) {
  std::fill(parents_.get(), parents_.get() + nelems_, -1);
}

template <typename T>
inline ArrayUnionFind<T>::ArrayUnionFind(ArrayUnionFind<T> const& other)
    : max_value_(other.max_value_),
      nelems_(other.nelems_),
      parents_(new SignedIntegral[nelems_]) {
  std::copy(other.parents_.get(), other.parents_.get() + nelems_,
            parents_.get());
}

template <typename T>
inline typename ArrayUnionFind<T>::SetIdType ArrayUnionFind<T>::Union(
    SetIdType s1,
    SetIdType s2) {
  ValueType set1_val = GetValue(s1);
  ValueType set2_val = GetValue(s2);
  parents_[s2] = s1;
  SetValue(s1, set1_val > set2_val ? set1_val : (set2_val + 1));
  return s1;
}

template <typename T>
inline typename ArrayUnionFind<T>::SetIdType ArrayUnionFind<T>::Find(
    ElemType elem) {
  auto root = std::as_const(*this).Find(elem);
  // Path compression
  while (elem != root) {
    auto prev = parents_[elem];
    parents_[elem] = root;
    elem = prev;
  }
  return static_cast<SetIdType>(root);
}

template <typename T>
inline typename ArrayUnionFind<T>::SetIdType ArrayUnionFind<T>::Find(
    ElemType elem) const {
#ifdef TD_CHECK_ARGS
  if (elem >= nelems_ || elem < 0)
    throw std::out_of_range("elem is out of range");
#endif
  auto root = static_cast<SignedIntegral>(elem);
  while (parents_[root] >= 0)
    root = parents_[root];
  return static_cast<SetIdType>(root);
}

template <typename T>
inline typename ArrayUnionFind<T>::ValueType ArrayUnionFind<T>::GetValue(
    SetIdType set_id) const {
#ifdef TD_CHECK_ARGS
  if (set_id >= nelems_ || set_id < 0)
    throw std::out_of_range("set_id is out of range");
  if (parents_[set_id] >= 0)
    throw std::invalid_argument("Argument is not a set id");
#endif
  return static_cast<ValueType>(-parents_[set_id]);
}

template <typename T>
inline typename ArrayUnionFind<T>::ValueType ArrayUnionFind<T>::GetMaxValue()
    const {
  return max_value_;
}

template <typename T>
inline void ArrayUnionFind<T>::SetValue(SetIdType set_id, ValueType value) {
  if (value > max_value_)
    max_value_ = static_cast<UnsignedIntegral>(value);
  parents_[set_id] = -static_cast<SignedIntegral>(value);
}
}  // namespace td
