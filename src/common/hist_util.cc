/*!
 * Copyright 2017-2020 by Contributors
 * \file hist_util.cc
 */
#include <dmlc/timer.h>
#include <dmlc/omp.h>

#include <rabit/rabit.h>
#include <numeric>
#include <vector>

#include "xgboost/base.h"
#include "../common/common.h"
#include "hist_util.h"
#include "hist_builder.h"
#include "random.h"
#include "column_matrix.h"
#include "quantile.h"
#include "../data/gradient_index.h"

namespace xgboost {
namespace common {

constexpr size_t Prefetch::kNoPrefetchSize;

HistogramCuts::HistogramCuts() {
  cut_ptrs_.HostVector().emplace_back(0);
}

/*!
 * \brief fill a histogram by zeros in range [begin, end)
 */
void InitilizeHistByZeroes(GHistRow hist, size_t begin, size_t end) {
#if defined(XGBOOST_STRICT_R_MODE) && XGBOOST_STRICT_R_MODE == 1
  std::fill(hist.begin() + begin, hist.begin() + end, xgboost::GradientPairPrecise());
#else  // defined(XGBOOST_STRICT_R_MODE) && XGBOOST_STRICT_R_MODE == 1
  memset(hist.data() + begin, '\0', (end - begin) * sizeof(xgboost::GradientPairPrecise));
#endif  // defined(XGBOOST_STRICT_R_MODE) && XGBOOST_STRICT_R_MODE == 1
}

/*!
 * \brief Increment hist as dst += add in range [begin, end)
 */
void IncrementHist(GHistRow dst, const GHistRow add, size_t begin, size_t end) {
  double* pdst = reinterpret_cast<double*>(dst.data());
  const double *padd = reinterpret_cast<const double *>(add.data());

  for (size_t i = 2 * begin; i < 2 * end; ++i) {
    pdst[i] += padd[i];
  }
}

/*!
 * \brief Copy hist from src to dst in range [begin, end)
 */
void CopyHist(GHistRow dst, const GHistRow src, size_t begin, size_t end) {
  double *pdst = reinterpret_cast<double *>(dst.data());
  const double *psrc = reinterpret_cast<const double *>(src.data());

  for (size_t i = 2 * begin; i < 2 * end; ++i) {
    pdst[i] = psrc[i];
  }
}

/*!
 * \brief Compute Subtraction: dst = src1 - src2 in range [begin, end)
 */
void SubtractionHist(GHistRow dst, const GHistRow src1, const GHistRow src2, size_t begin,
                     size_t end) {
  double* pdst = reinterpret_cast<double*>(dst.data());
  const double* psrc1 = reinterpret_cast<const double*>(src1.data());
  const double* psrc2 = reinterpret_cast<const double*>(src2.data());

  for (size_t i = 2 * begin; i < 2 * end; ++i) {
    pdst[i] = psrc1[i] - psrc2[i];
  }
}

// template<typename GradientSumT>
void ClearHist(double* dest_hist,
                size_t begin, size_t end) {
  for (size_t bin_id = begin; bin_id < end; ++bin_id) {
    dest_hist[bin_id]  = 0;
  }
}
// template void ClearHist(float* dest_hist,
//                         size_t begin, size_t end);
// template void ClearHist(double* dest_hist,
//                         size_t begin, size_t end);

// template<typename GradientSumT>
void ReduceHist(double* dest_hist,
                const std::vector<std::vector<uint16_t>>& local_threads_mapping,
                std::vector<std::vector<std::vector<double>>>* histograms,
                const size_t node_id,
                const std::vector<uint16_t>& threads_id_for_node,
                size_t begin, size_t end) {
  const size_t first_thread_id = threads_id_for_node[0];
  CHECK_LT(node_id, local_threads_mapping[first_thread_id].size());
  const size_t mapped_nod_id = local_threads_mapping[first_thread_id][node_id];
  CHECK_LT(mapped_nod_id, (*histograms)[first_thread_id].size());
  double* hist0 =  (*histograms)[first_thread_id][mapped_nod_id].data();

  for (size_t bin_id = begin; bin_id < end; ++bin_id) {
    dest_hist[bin_id] = hist0[bin_id];
    hist0[bin_id] = 0;
  }
  for (size_t tid = 1; tid < threads_id_for_node.size(); ++tid) {
    const size_t thread_id = threads_id_for_node[tid];
    const size_t mapped_nod_id = local_threads_mapping[thread_id][node_id];
    double* hist =  (*histograms)[thread_id][mapped_nod_id].data();
    for (size_t bin_id = begin; bin_id < end; ++bin_id) {
      dest_hist[bin_id] += hist[bin_id];
      hist[bin_id] = 0;
    }
  }
}

// template void ReduceHist(float* dest_hist,
//                          const std::vector<std::vector<uint16_t>>& local_threads_mapping,
//                          std::vector<std::vector<std::vector<float>>>* histograms,
//                          const size_t node_displace,
//                          const std::vector<uint16_t>& threads_id_for_node,
//                          size_t begin, size_t end);
// template void ReduceHist(double* dest_hist,
//                          const std::vector<std::vector<uint16_t>>& local_threads_mapping,
//                          std::vector<std::vector<std::vector<double>>>* histograms,
//                          const size_t node_displace,
//                          const std::vector<uint16_t>& threads_id_for_node,
//                          size_t begin, size_t end);

}  // namespace common
}  // namespace xgboost
