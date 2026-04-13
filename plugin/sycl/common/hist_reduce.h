#ifndef PLUGIN_SYCL_COMMON_HIST_REDUCE_H_
#define PLUGIN_SYCL_COMMON_HIST_REDUCE_H_

#include <CL/sycl.hpp>
#include "../data.h"
#include "../../src/common/hist_util.h"

namespace xgboost {
namespace sycl {
namespace common {

::sycl::event ReduceHistParallel(::sycl::queue* qu,
                                 GradientPairInt64* hist_data,
                                 GradientPairInt64* hist_buffer_data,
                                 size_t nblocks, size_t nbins,
                                 const ::sycl::event& event_main);

}  // namespace common
}  // namespace sycl
}  // namespace xgboost

#endif  // PLUGIN_SYCL_COMMON_HIST_REDUCE_H_
