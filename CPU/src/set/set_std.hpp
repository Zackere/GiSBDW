#pragma once
#include <set>
#include "set_base.hpp"

namespace td {
template <class UnsignedIntegral>
class SetStd : public SetBase<UnsignedIntegral> {
  // typedef typename UnsignedIntegral Element;
  using Element = UnsignedIntegral;

 public:
  SetStd() = delete;
  SetStd(Element maximumSize) : SetBase<UnsignedIntegral>(maximumSize){};
  virtual void Add(Element x) override {
    set_.insert(x);
    this->numberOfElements_++;
  }
  virtual void Remove(Element x) override {
    set_.erase(set_.find(x));
    this->numberOfElements_--;
  }

  virtual size_t Encode() override { return this->EncodeGeneric(); }

  virtual void Decode(size_t code, Element n, Element k) override {
    this->DecodeGeneric(code, n, k);
  }
  virtual bool Contains(Element x) override {
    return set_.find(x) != set_.end();
  }
  virtual void Clear() override { set_.clear(); }

  virtual Element GetElementAtIndex(Element index) {
    return *(set_begin() + index);
  }

  virtual ~SetStd() override = default;

 private:
  std::set<UnsignedIntegral> set_;
};
}  // namespace td
