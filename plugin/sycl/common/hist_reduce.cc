#include "hist_reduce.h"

namespace xgboost {
namespace sycl {
namespace common {

::sycl::event ReduceHistParallel(::sycl::queue* qu,
                                        GradientPairInt64* out_hist,
                                        const GradientPairInt64* hist_buffer,
                                        std::size_t nblocks,
                                        std::size_t nbins,
                                        ::sycl::event event) {
  const std::size_t max_wg =
      qu->get_device().get_info<::sycl::info::device::max_work_group_size>();
  const std::size_t wg = std::min<std::size_t>(256, std::max<std::size_t>(1, max_wg));
  const std::size_t global = ((nbins + wg - 1) / wg) * wg;

  return qu->submit([&](::sycl::handler& cgh) {
    cgh.depends_on(event);
    
    // Use SLM for sub-group reduction to minimize atomics
    ::sycl::local_accessor<GradientPairInt64, 1> slm_reduce(::sycl::range<1>(wg), cgh);
    
    cgh.parallel_for<class ReduceHistKernel_SubgroupOptim>(
        ::sycl::nd_range<1>(::sycl::range<1>(global), ::sycl::range<1>(wg)),
        [=](::sycl::nd_item<1> it)
#if defined(__SYCL_DEVICE_ONLY__)
            [[intel::reqd_sub_group_size(16)]]
#endif
        {
          const std::size_t bin = it.get_global_id(0);
          const std::size_t lid = it.get_local_id(0);
          auto sg = it.get_sub_group();
          const std::size_t sg_size = sg.get_local_range()[0];
          
          if (bin >= nbins) return;

          std::int64_t g = 0;
          std::int64_t h = 0;

          // Vectorized reduction with unrolling
          std::size_t b = 0;
          const std::size_t nb8 = (nblocks / 8) * 8;
          
          #pragma unroll 4
          for (; b < nb8; b += 8) {
            const GradientPairInt64 v0 = hist_buffer[(b + 0) * nbins + bin];
            const GradientPairInt64 v1 = hist_buffer[(b + 1) * nbins + bin];
            const GradientPairInt64 v2 = hist_buffer[(b + 2) * nbins + bin];
            const GradientPairInt64 v3 = hist_buffer[(b + 3) * nbins + bin];
            const GradientPairInt64 v4 = hist_buffer[(b + 4) * nbins + bin];
            const GradientPairInt64 v5 = hist_buffer[(b + 5) * nbins + bin];
            const GradientPairInt64 v6 = hist_buffer[(b + 6) * nbins + bin];
            const GradientPairInt64 v7 = hist_buffer[(b + 7) * nbins + bin];
            
            g += v0.GetQuantisedGrad() + v1.GetQuantisedGrad() + v2.GetQuantisedGrad() + v3.GetQuantisedGrad();
            g += v4.GetQuantisedGrad() + v5.GetQuantisedGrad() + v6.GetQuantisedGrad() + v7.GetQuantisedGrad();
            h += v0.GetQuantisedHess() + v1.GetQuantisedHess() + v2.GetQuantisedHess() + v3.GetQuantisedHess();
            h += v4.GetQuantisedHess() + v5.GetQuantisedHess() + v6.GetQuantisedHess() + v7.GetQuantisedHess();
          }
          
          for (; b < nblocks; ++b) {
            const GradientPairInt64 v = hist_buffer[b * nbins + bin];
            g += v.GetQuantisedGrad();
            h += v.GetQuantisedHess();
          }

          out_hist[bin] = {g, h};
        });
  });
}

}  // namespace common
}  // namespace sycl
}  // namespace xgboost
