/*!
 * Copyright 2025 by Contributors
 * \file gradient_quantiser.h
 * \brief Quantises float gradients to int64 for faster atomic histogram accumulation.
 *        Follows the same algorithm as the CUDA path (Algorithm 5 from Demmel & Nguyen).
 */
#ifndef PLUGIN_SYCL_TREE_GRADIENT_QUANTISER_H_
#define PLUGIN_SYCL_TREE_GRADIENT_QUANTISER_H_

#include <cmath>
#include <limits>
#include <algorithm>
#include <array>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtautological-constant-compare"
#pragma GCC diagnostic ignored "-W#pragma-messages"
#include <xgboost/base.h>
#pragma GCC diagnostic pop

namespace xgboost {
namespace sycl {
namespace tree {

class GradientQuantiser {
  double to_fixed_point_grad_;
  double to_fixed_point_hess_;
  double to_floating_point_grad_;
  double to_floating_point_hess_;

  template <typename T>
  static T CreateRoundingFactor(T max_abs, size_t n) {
    T delta = max_abs / (T(1.0) - T(2.0) * T(n) * std::numeric_limits<T>::epsilon());
    int exp;
    std::frexp(delta, &exp);
    return std::ldexp(T(1.0), exp);
  }

 public:
  GradientQuantiser()
    : to_fixed_point_grad_(0), to_fixed_point_hess_(0),
      to_floating_point_grad_(0), to_floating_point_hess_(0) {}

  void Configure(::sycl::queue* qu, const GradientPair* gpair, size_t n_rows) {
    double pos_grad = 0, neg_grad = 0, pos_hess = 0, neg_hess = 0;
    {
      size_t block_size = 32;
      size_t n_blocks = n_rows / block_size + (n_rows % block_size > 0);

      ::sycl::buffer<double, 1> pg_buf(&pos_grad, 1);
      ::sycl::buffer<double, 1> ng_buf(&neg_grad, 1);
      ::sycl::buffer<double, 1> ph_buf(&pos_hess, 1);
      ::sycl::buffer<double, 1> nh_buf(&neg_hess, 1);
      qu->submit([&](::sycl::handler& cgh) {
        auto pg_red = ::sycl::reduction(pg_buf, cgh, ::sycl::plus<double>());
        auto ng_red = ::sycl::reduction(ng_buf, cgh, ::sycl::plus<double>());
        auto ph_red = ::sycl::reduction(ph_buf, cgh, ::sycl::plus<double>());
        auto nh_red = ::sycl::reduction(nh_buf, cgh, ::sycl::plus<double>());
        cgh.parallel_for<>(::sycl::range<1>(n_blocks),
                           pg_red, ng_red, ph_red, nh_red,
                           [=](::sycl::item<1> pid, auto& pg, auto& ng,
                               auto& ph, auto& nh) {
          size_t block = pid.get_id(0);

          double pg_local = 0;
          double ng_local = 0;
          double ph_local = 0;
          double nh_local = 0;

          size_t begin = block * block_size;
          size_t end = std::min<size_t>(begin + block_size, n_rows);
          for (size_t i = begin; i < end; ++i) {
            auto grad = static_cast<double>(gpair[i].GetGrad());
            auto hess = static_cast<double>(gpair[i].GetHess());
            if (grad > 0) pg_local += grad; else ng_local += (-grad);
            if (hess > 0) ph_local += hess; else nh_local += (-hess);
          }
          pg += pg_local;
          ng += ng_local;
          ph += ph_local;
          nh += nh_local;

        });
      }).wait();
    }

    double max_grad = std::max(pos_grad, neg_grad);
    double max_hess = std::max(pos_hess, neg_hess);

    double rounding_grad = CreateRoundingFactor<double>(max_grad, n_rows);
    double rounding_hess = CreateRoundingFactor<double>(max_hess, n_rows);

    // Use 62 bits of precision (64 - 1 sign - 1 overflow guard)
    constexpr double kShift = static_cast<double>(static_cast<int64_t>(1) << 62);
    to_floating_point_grad_ = rounding_grad / kShift;
    to_floating_point_hess_ = rounding_hess / kShift;
    to_fixed_point_grad_ = 1.0 / to_floating_point_grad_;
    to_fixed_point_hess_ = 1.0 / to_floating_point_hess_;
  }

  GradientPairInt64 ToFixedPoint(const GradientPair& gpair) const {
    return GradientPairInt64(
      static_cast<int64_t>(static_cast<double>(gpair.GetGrad()) * to_fixed_point_grad_),
      static_cast<int64_t>(static_cast<double>(gpair.GetHess()) * to_fixed_point_hess_));
  }

  GradientPairPrecise ToFloatingPoint(const GradientPairInt64& gpair) const {
    return GradientPairPrecise(
      gpair.GetQuantisedGrad() * to_floating_point_grad_,
      gpair.GetQuantisedHess() * to_floating_point_hess_);
  }

  double GetFixedPointGrad() const { return to_fixed_point_grad_; }
  double GetFixedPointHess() const { return to_fixed_point_hess_; }
  double GetFloatingPointGrad() const { return to_floating_point_grad_; }
  double GetFloatingPointHess() const { return to_floating_point_hess_; }
};

}  // namespace tree
}  // namespace sycl
}  // namespace xgboost

#endif  // PLUGIN_SYCL_TREE_GRADIENT_QUANTISER_H_
