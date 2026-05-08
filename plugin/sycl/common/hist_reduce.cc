#include "hist_reduce.h"

namespace xgboost {
namespace sycl {
namespace common {

::sycl::event ReduceHistParallel(::sycl::queue* qu,
                                 GradientPairInt64* hist_data,
                                 GradientPairInt64* hist_buffer_data,
                                 size_t nblocks, size_t nbins,
                                 const ::sycl::event& event) {
  constexpr size_t kGroupSize = 64;
  size_t n_groups = nblocks / kGroupSize + (nblocks % kGroupSize > 0);

  int64_t* hist = reinterpret_cast<int64_t*>(hist_data);
  auto event_init = qu->fill(hist, int64_t(0), 2 * nbins, event);

  auto event_save = qu->submit([&](::sycl::handler& cgh) {
    cgh.depends_on(event_init);
    cgh.parallel_for<>(::sycl::range<2>(n_groups, nbins),
                        [=](::sycl::item<2> pid) {
      size_t group = pid.get_id(0);
      size_t idx_bin = pid.get_id(1);

      size_t begin = group * kGroupSize;
      size_t end = begin + kGroupSize;
      if (end > nblocks) end = nblocks;

      GradientPairInt64 gpair(0, 0);
      for (size_t j = begin; j < end; ++j) {
        gpair += hist_buffer_data[j * nbins + idx_bin];
      }

      AtomicRef<int64_t> grad(hist[2 * idx_bin]);
      AtomicRef<int64_t> hess(hist[2 * idx_bin + 1]);
      grad += gpair.GetQuantisedGrad();
      hess += gpair.GetQuantisedHess();
    });
  });

  return event_save;
}

}  // namespace common
}  // namespace sycl
}  // namespace xgboost
