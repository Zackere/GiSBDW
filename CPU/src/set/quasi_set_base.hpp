#pragma once

namespace td {
template <class UnsignedIntegral>
class QuasiSetBase {
 public:
  using Element = UnsignedIntegral;
  QuasiSetBase() = delete;
  QuasiSetBase(Element maxSize) : maxSize_(maxSize), numberOfElements_(0) {}
  virtual ~QuasiSetBase() = default;

  virtual void ExcludeTemporarilyElementAtIndex(Element index) = 0;
  virtual void RecoverExcludedElement() = 0;
  virtual void Decode(size_t code, Element k) = 0;
  virtual size_t EncodeExcluded() = 0;
  virtual Element GetElementAtIndex(Element index) = 0;
  Element GetNumberOfElements() { return numberOfElements_; }
  Element GetMaximumSize() { return maxSize_; }

 protected:
  Element maxSize_;
  Element numberOfElements_;
};
}  // namespace td
