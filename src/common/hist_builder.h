/*!
 * Copyright 2017-2021 by Contributors
 * \file hist_builder.h
 */
#ifndef XGBOOST_COMMON_HIST_BUILDER_H_
#define XGBOOST_COMMON_HIST_BUILDER_H_

#include <algorithm>
#include <vector>
#include "hist_util.h"
#include "../data/gradient_index.h"
#include "column_matrix.h"

#if defined(XGBOOST_MM_PREFETCH_PRESENT)
  #include <xmmintrin.h>
  #define PREFETCH_READ_T0(addr) _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_T0)
#elif defined(XGBOOST_BUILTIN_PREFETCH_PRESENT)
  #define PREFETCH_READ_T0(addr) __builtin_prefetch(reinterpret_cast<const char*>(addr), 0, 3)
#else  // no SW pre-fetching available; PREFETCH_READ_T0 is no-op
  #define PREFETCH_READ_T0(addr) do {} while (0)
#endif  // defined(XGBOOST_MM_PREFETCH_PRESENT)

namespace xgboost {
namespace common {

struct Prefetch {
 public:
  static constexpr size_t kCacheLineSize = 64;
  static constexpr size_t kPrefetchOffset = 10;

 private:
  static constexpr size_t kNoPrefetchSize =
      kPrefetchOffset + kCacheLineSize /
      sizeof(decltype(GHistIndexMatrix::row_ptr)::value_type);

 public:
  static size_t NoPrefetchSize(size_t rows) {
    return std::min(rows, kNoPrefetchSize);
  }

  template <typename T>
  static constexpr size_t GetPrefetchStep() {
    return Prefetch::kCacheLineSize / sizeof(T);
  }
};

template<bool do_prefetch,
         typename BinIdxType,
         bool feature_blocking,
         bool is_root,
         bool any_missing,
         typename GradientSumT>
inline void RowsWiseBuildHist(const BinIdxType* gradient_index,
                              const uint32_t* rows,
                              const size_t* row_ptr,
                              const uint32_t row_begin,
                              const uint32_t row_end,
                              const size_t n_features,
                              uint16_t* nodes_ids,
                              const std::vector<std::vector<uint64_t>>& hists_addr,
                              const uint16_t* mapping_ids,
                              const float* pgh, const size_t ib) {
  const uint32_t two {2};
  for (size_t ri = row_begin; ri < row_end; ++ri) {
    const size_t i = is_root ? ri : rows[ri];
    const size_t icol_start = any_missing ? row_ptr[i] : i * n_features;
    const size_t icol_end = any_missing ? row_ptr[i + 1] : icol_start + n_features;
    const size_t row_sizes = any_missing ? icol_end - icol_start : n_features;
    const BinIdxType* gr_index_local = gradient_index + icol_start;
    const size_t idx_gh = two * i;

    const uint32_t nid = is_root ? 0 : mapping_ids[nodes_ids[i]];
    if (do_prefetch) {
      const size_t icol_start_prefetch = any_missing ?
                                         row_ptr[rows[ri + common::Prefetch::kPrefetchOffset]] :
                                         rows[ri + common::Prefetch::kPrefetchOffset] * n_features;
      const size_t icol_end_prefetch = any_missing ?
                                       row_ptr[rows[ri + common::Prefetch::kPrefetchOffset] + 1] :
                                       (icol_start_prefetch + n_features);

      PREFETCH_READ_T0(pgh + two * rows[ri + common::Prefetch::kPrefetchOffset]);
      PREFETCH_READ_T0(0 + rows[ri + common::Prefetch::kPrefetchOffset]);
      for (size_t j = icol_start_prefetch; j < icol_end_prefetch;
          j += common::Prefetch::GetPrefetchStep<BinIdxType>()) {
        PREFETCH_READ_T0(gradient_index + j);
      }
    } else if (is_root) {
      nodes_ids[i] = 0;
    }
    const uint64_t* offsets64 = hists_addr[nid].data();
    const size_t begin = feature_blocking ? ib*13 : 0;
    const size_t end = feature_blocking ? std::min(begin + 13, n_features) : row_sizes;

    // The hint with pgh_t helps the compiler to use more effective processor instructions.
    const GradientSumT pgh_t[] = {pgh[idx_gh], pgh[idx_gh + 1]};
    for (size_t jb = begin;  jb < end; ++jb) {
      GradientSumT* hist_local = reinterpret_cast<GradientSumT*> (
        offsets64[jb] + (static_cast<size_t>(gr_index_local[jb])) * 2 * sizeof(GradientSumT));
      *(hist_local)     +=  pgh_t[0];
      *(hist_local + 1) +=  pgh_t[1];
    }
  }
}

template<typename BinIdxType,
         bool is_root, bool any_missing,
         bool column_sampling, typename GradientSumT>
void ColWiseBuildHist(const std::vector<GradientPair>& gpair,
                      const uint32_t* rows,
                      const uint32_t row_begin,
                      const uint32_t row_end,
                      const size_t n_features,
                      uint16_t* nodes_ids,
                      const std::vector<std::vector<uint64_t>>& hists_addr,
                      const common::ColumnMatrix& column_matrix,
                      const uint16_t* mapping_ids,
                      const std::vector<int>& fids) {
  using DenseColumnT = common::DenseColumn<BinIdxType, any_missing>;
  const float* pgh = reinterpret_cast<const float*>(gpair.data());
  const size_t n_columns = column_sampling ? fids.size() : n_features;
  for (size_t cid = 0; cid < n_columns; ++cid) {
    const size_t local_cid = column_sampling ? fids[cid] : cid;
    const auto column_ptr = column_matrix.GetColumn<BinIdxType, any_missing>(local_cid);
    const DenseColumnT* column = dynamic_cast<const DenseColumnT*>(column_ptr.get());
    CHECK_NE(column, static_cast<const DenseColumnT*>(nullptr));
    const BinIdxType* gr_index_local = column->GetFeatureBinIdxPtr().data();
    const size_t base_idx = column->GetBaseIdx();
    for (size_t ii = row_begin; ii < row_end; ++ii) {
      const size_t row_id = is_root ? ii : rows[ii];
      if (is_root && (cid == 0)) {
        nodes_ids[row_id] = 0;
      }
      if (!any_missing || (any_missing && !column->IsMissing(row_id))) {
        const uint32_t nid = is_root ? 0 : mapping_ids[nodes_ids[row_id]];
        const size_t idx_gh = row_id << 1;
        const uint64_t* offsets64 = hists_addr[nid].data();

        // The hint with pgh_t helps the compiler to use more effective processor instructions.
        const GradientSumT pgh_t[2] = {pgh[idx_gh], pgh[idx_gh + 1]};
        GradientSumT* hist_local = reinterpret_cast<GradientSumT*> (
          offsets64[local_cid] + (static_cast<size_t>(gr_index_local[row_id])) * 2 * sizeof(GradientSumT) + 
                                 (any_missing ? base_idx * 2 * sizeof(GradientSumT) : 0));
        *(hist_local)     +=  pgh_t[0];
        *(hist_local + 1) +=  pgh_t[1];
      }
    }
  }
}

template<typename GradientSumT, typename BinIdxType,
         bool read_by_column, bool feature_blocking,
         bool is_root, bool any_missing, bool column_sampling>
void BuildHist(const std::vector<GradientPair>& gpair,
               const uint32_t* rows,
               const uint32_t row_begin,
               const uint32_t row_end,
               const GHistIndexMatrix& gmat,
               const size_t n_features, const BinIdxType* numa,
               uint16_t* nodes_ids, const std::vector<std::vector<uint64_t>>& hists_addr,
               const common::ColumnMatrix& column_matrix,
               const uint16_t* mapping_ids,
               const std::vector<int>& fids) {
  if (read_by_column) {
    ColWiseBuildHist<BinIdxType,
                     is_root, any_missing,
                     column_sampling,
                     GradientSumT> (gpair, rows, row_begin,
                                    row_end, n_features,
                                    nodes_ids, hists_addr,
                                    column_matrix,
                                    mapping_ids, fids);
  } else {
    const size_t row_size = row_end - row_begin;
    const size_t* row_ptr =  gmat.row_ptr.data();
    const float* pgh = reinterpret_cast<const float*>(gpair.data());
    // TODO(ShvetsKS): check numa enabling for sparse
    const BinIdxType* gradient_index = any_missing ? gmat.index.data<BinIdxType>() : numa;

    const size_t feature_block_size = feature_blocking ? static_cast<size_t>(13) : n_features;
    const size_t nb = n_features / feature_block_size + !!(n_features % feature_block_size);   // NOLINT
    const size_t size_with_prefetch = (is_root || feature_blocking) ? 0 :
                                      ((row_size > common::Prefetch::kPrefetchOffset) ?
                                      (row_size - common::Prefetch::kPrefetchOffset) : 0);
    for (size_t ib = 0; ib < nb; ++ib) {
      RowsWiseBuildHist<true, BinIdxType,
                        feature_blocking,
                        is_root, any_missing,
                        GradientSumT> (gradient_index, rows, row_ptr, row_begin,
                                    row_begin + size_with_prefetch,
                                    n_features, nodes_ids,
                                    hists_addr, mapping_ids, pgh, ib);
      RowsWiseBuildHist<false, BinIdxType,
                        feature_blocking,
                        is_root, any_missing,
                        GradientSumT> (gradient_index, rows, row_ptr, row_begin + size_with_prefetch,
                                    row_end, n_features, nodes_ids,
                                    hists_addr, mapping_ids, pgh, ib);
    }
  }
}

}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_HIST_BUILDER_H_
