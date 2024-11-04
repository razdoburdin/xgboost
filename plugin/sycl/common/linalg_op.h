/**
 * Copyright 2021-2024, XGBoost Contributors
 * \file linalg_op.h
 */
#ifndef PLUGIN_SYCL_COMMON_LINALG_OP_H_
#define PLUGIN_SYCL_COMMON_LINALG_OP_H_

#include <vector>
#include <utility>

// #include "../../../src/common/linalg_op.h"

#include "../data.h"
#include "../device_manager.h"

#include <CL/sycl.hpp>

namespace xgboost {
namespace sycl {
namespace linalg {

template<typename T, std::int32_t D>
using TensorView = xgboost::linalg::TensorView<T, D>;

struct WorkGroupsParams {
  size_t n_workgroups;
  size_t workgroup_size;
};

template <typename Fn>
::sycl::event GroupWiseKernel(::sycl::queue* qu, int* flag_ptr,
                              const std::vector<::sycl::event>& events,
                              const WorkGroupsParams& wg, Fn &&fn) {
  ::sycl::buffer<int, 1> flag_buf(flag_ptr, 1);
  auto event = qu->submit([&](::sycl::handler& cgh) {
    cgh.depends_on(events);
    auto flag  = flag_buf.get_access<::sycl::access::mode::write>(cgh);
    cgh.parallel_for_work_group<>(::sycl::range<1>(wg.n_workgroups),
                                  ::sycl::range<1>(wg.workgroup_size),
                                  [=](::sycl::group<1> group) {
      group.parallel_for_work_item([&](::sycl::h_item<1> item) {
        const size_t idx = item.get_global_id()[0];
        fn(idx, flag);
      });
    });
  });
  return event;
}

// // Use template specialization to dispatch, Windows + CUDA 11.8 doesn't support extended
// // lambda inside constexpr if
// template <typename T, std::int32_t D>
// struct ElementWiseImpl {
//   template <typename Fn>
//   void operator()(TensorView<T, D> t, Fn&& fn, cudaStream_t s) {
//     static_assert(D > 1);
//     dh::LaunchN(t.Size(), s, [=] __device__(std::size_t i) mutable {
//       std::apply(fn, linalg::UnravelIndex(i, t.Shape()));
//     });
//   }
// };

// template <typename T>
// struct ElementWiseImpl<T, 1> {
//   template <typename Fn>
//   void operator()(TensorView<T, 1> t, Fn&& fn, cudaStream_t s) {
//     dh::LaunchN(t.Size(), s, [=] __device__(std::size_t i) { fn(i); });
//   }
// };

template<typename Fn, typename TupleType, size_t ... I>
auto call(Fn&& fn, TupleType t, std::index_sequence<I ...>) {
     return fn(std::get<I>(t) ...);
}

template<typename Fn, typename TupleType>
auto call(Fn&& fn, TupleType t) {
    static constexpr auto size = std::tuple_size<TupleType>::value;
    return call(fn, t, std::make_index_sequence<size>{});
}

template <typename T, std::int32_t D, typename Fn>
void ElementWiseKernel(TensorView<T, D> t, Fn&& fn) {
  sycl::DeviceManager device_manager;
  auto* qu = device_manager.GetQueue(t.Device());
  qu->submit([&](::sycl::handler& cgh) {
    cgh.parallel_for<>(::sycl::range<1>(t.Size()),
                       [=](::sycl::id<1> pid) {
      const size_t idx = pid[0];
      // call(fn, xgboost::linalg::UnravelIndex(idx, t.Shape()));
    });
  }).wait_and_throw();
}

}  // namespace linalg
}  // namespace sycl
}  // namespace xgboost
#endif  // PLUGIN_SYCL_COMMON_LINALG_OP_H_
