#pragma once
#include <memory>

#include "../binomial_coefficients/binomial_coefficient.hpp"
#include "set_base.hpp"

namespace td {

template <class UnsignedIntegral>
class SetArray : public SetBase<UnsignedIntegral> {
 public:
  using Element = UnsignedIntegral;

  SetArray(Element maximumSize)
      : SetBase<UnsignedIntegral>(maximumSize),
        arr_(new Element[maximumSize]){};

  SetArray() = delete;
  virtual ~SetArray() override = default;

  virtual void Add(Element x) override {
    arr_[this->numberOfElements_] = x;
    this->numberOfElements_++;
  }
  virtual void Remove(Element x) override { return; }

  virtual bool Contains(Element x) override {
    for (Element i = 0; i < this->GetMaximumSize(); ++i) {
      if (arr_[i] == x)
        return true;
    }
    return false;
  }
  virtual void Clear() override { this->numberOfElements_ = 0; }
  virtual Element GetElementAtIndex(Element index) override {
    return arr_[index];
  }
  virtual void Decode(size_t code, Element n, Element k) override {
    while (k > 0) {
      --n;
      size_t nk = NChooseK(n, k);
      if (code > nk) {
        k--;
        arr_[k] = n;
        code -= nk;
      }
    }
    this->numberOfElements_ = k;
  }

  virtual size_t Encode() override {
    size_t code = 0;
    for (UnsignedIntegral i = 0; i < this->GetNumberOfElements(); ++i) {
      code += NChooseK(arr_[i], i + 1);
    }
    return code;
  }

 private:
  std::unique_ptr<Element[]> arr_;
};
}  // namespace td
