#pragma once
#include <boost/math/special_functions/binomial.hpp>
#include "../binomial_coefficients/binomial_coefficient.hpp"

namespace td
{
	template <class UnsignedIntegral>
	class SetBase
	{
		//typedef typename UnsignedIntegral Element;
		using Element = UnsignedIntegral;
	public:
		SetBase() = delete;
		SetBase(Element maximumSize) :maximumSize_(maximumSize), numberOfElements_(0) {};
		virtual ~SetBase() = default;
		virtual void Add(Element x) = 0;
		virtual void Remove(Element x) = 0;
		virtual size_t Encode() = 0;
		virtual void Decode(size_t code, Element n, Element k) = 0;
		virtual bool Contains(Element x) = 0;
		virtual void Clear() = 0;
		virtual Element GetElementAtIndex(Element index) = 0;
		Element GetNumberOfElements() { return numberOfElements_; }
		Element GetMaximumSize() { return maximumSize_; }
		size_t EncodeGeneric();
		void DecodeGeneric(size_t code, Element n, Element k);
	protected:
		Element const maximumSize_;
		Element numberOfElements_;
	};


	template<class UnsignedIntegral>
	inline size_t SetBase<UnsignedIntegral>::EncodeGeneric()
	{
		size_t code = 0;
		Element k = GetNumberOfElements();
		Element n = GetMaximumSize();
		while (k > 0)
		{
			--n;
			if (Contains(n)) {
				code += NChooseK(n, k);
				--k;
			}
		}
		return code;
	}

	template<class UnsignedIntegral>
	inline void SetBase<UnsignedIntegral>::DecodeGeneric(size_t code, Element n, Element k)
	{
		Clear();
		while (k > 0)
		{
			--n;
			size_t nk = NChooseK(n, k);
			if (code >= nk)
			{
				Add(n);
				code -= nk;
				--k;
			}
		}
	}
}
