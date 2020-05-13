// Copyright 2020 GISBDW. All rights reserved.
#pragma once

#include <utility>
#include <vector>
#include "../binomial_coefficients/binomial_coefficient.hpp"
#include "quasi_set_base.hpp"

#ifdef TD_CHECK_ARGS
#include <stdexcept>
#endif

namespace td {

template <class UnsignedIntegral>
class QuasiSetArray : public QuasiSetBase<UnsignedIntegral> {
 public:
  using Element = UnsignedIntegral;

  QuasiSetArray() = delete;
  explicit QuasiSetArray(Element maxSize)
      : QuasiSetBase<UnsignedIntegral>(maxSize),
        arr_(maxSize),
        excludedIndex_(0) {
#ifdef TD_CHECK_ARGS
    isElementExcluded_ = false;
#endif
  }

  ~QuasiSetArray() override = default;

  void ExcludeTemporarilyElementAtIndex(Element index) override {
#ifdef TD_CHECK_ARGS
    if (isElementExcluded_)
      throw std::logic_error(
          "Only one element can be excluded at the same time");
    if (index > this->numberOfElements_)
      throw std::out_of_range(
          "Index was bigger than number of elements in container");
    isElementExcluded_ = true;
#endif
    excludedIndex_ = index;
    this->numberOfElements_--;
    std::swap(arr_[index], arr_[this->numberOfElements_]);
  }

  void RecoverExcludedElement() override {
#ifdef TD_CHECK_ARGS
    if (!isElementExcluded_)
      throw std::logic_error(
          "Cannot recover excluded element, because no element was excluded");
    isElementExcluded_ = false;
#endif
    std::swap(arr_[excludedIndex_], arr_[this->numberOfElements_]);
    this->numberOfElements_++;
  }

  Element GetElementAtIndex(Element index) override { return arr_[index]; }

  void Decode(size_t code, Element k) override {
    numberOfElements_ = k;
    auto n = GetMaximumSize();
    while (k > 0) {
      --n;
      size_t nk = NChooseK(n, k);
      if (code >= nk) {
        --k;
        arr_[k] = n;
        code -= nk;
      }
    }
  }
  size_t EncodeExcluded() override {
#ifdef TD_CHECK_ARGS
    if (!isElementExcluded_)
      throw std::logic_error(
          "Call to EncodeExcluded when no element was excluded");
#endif
    size_t code = 0;
    Element k = 1;
    for (Element i = 0; i < excludedIndex_; ++i) {
      code += NChooseK(arr_[i], k);
      k++;
    }
    for (Element i = excludedIndex_ + 1; i < this->numberOfElements_; ++i) {
      code += NChooseK(arr_[i], k);
      k++;
    }
    code += NChooseK(arr_[excludedIndex_], k);
    return code;
  }

 private:
  std::vector<Element> arr_;
  Element excludedIndex_;
#ifdef TD_CHECK_ARGS
  bool isElementExcluded_;
#endif
};
}  // namespace td
