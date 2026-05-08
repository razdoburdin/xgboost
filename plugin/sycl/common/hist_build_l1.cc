/*!
 * Copyright 2017-2023 by Contributors
 * \file hist_build_l1.cc
 * Per-feature L1 histogram kernel in a separate TU
 * to avoid SYCL compiler cross-kernel interference.
 */
#include <vector>
#include <algorithm>

#include "../data/gradient_index.h"
#include "../tree/hist_dispatcher.h"
#include "hist_util.h"
#include "hist_reduce.h"

#include <sycl/sycl.hpp>

namespace xgboost {
namespace sycl {
namespace common {

::sycl::event BuildHistKernelSLM(::sycl::queue* qu,
                              const GradientPairInt64* pgh,
                              const std::vector<bst_node_t>& nodes,
                              RowSetCollection* row_set,
                              const GHistIndexMatrix& gmat,
                              HistCollectionInt64* histograms,
                              GHistRowInt64* hist_buffer,
                              const DeviceProperties& device_prop,
                              ::sycl::event event) {
  using BinIdxType = uint8_t;
  const size_t n_columns = gmat.nfeatures;
  const BinIdxType* gradient_index = gmat.index.data<BinIdxType>();
  const uint32_t* offsets = gmat.cut.cut_ptrs_.ConstDevicePointer();
  const size_t nbins = gmat.nbins;
  size_t n_nodes = nodes.size();

  ::sycl::event event_batch = event;

  size_t max_buffer_size = device_prop.local_mem_size / (sizeof(GradientPairInt64) * device_prop.eu_per_core);
  size_t bins_block = 0;
  size_t feat_in_block = 0;
  size_t buffer_size = 0;
  size_t work_group_size = n_columns;
  std::vector<bst_feature_t> foffsets(2, 0);
  for (size_t fid = 0; fid < n_columns; ++fid) {
    size_t nbins_feature = gmat.cut.cut_ptrs_.ConstHostVector()[fid + 1]
                         - gmat.cut.cut_ptrs_.ConstHostVector()[fid];
    if ((bins_block + nbins_feature < max_buffer_size) &&
        (feat_in_block < device_prop.max_sub_group_size)) {
      bins_block += nbins_feature;
      feat_in_block += 1;
      foffsets.back() = fid + 1;
    } else {
      work_group_size = std::min(work_group_size, feat_in_block);
      buffer_size = std::max(buffer_size, bins_block);
      bins_block = nbins_feature;
      feat_in_block = 1;
      foffsets.push_back(fid + 1);
    }
  }
  USMVector<bst_feature_t> foffsets_device(qu, foffsets);
  const auto* foffsets_ptr = foffsets_device.DataConst();
  size_t n_feature_groups = foffsets.size() - 1;

  GradientPairInt64* hist_buffer_data = hist_buffer->Data();
  size_t max_nblocks = hist_buffer->Size() / nbins;

  for (size_t nidx = 0; nidx < n_nodes; ++nidx) {
    bst_node_t nid = nodes[nidx];
    size_t n_rows = (*row_set)[nid].Size();
    if (n_rows == 0) continue;

    size_t nblocks = max_nblocks;
    size_t block_size = n_rows / nblocks + (n_rows % nblocks > 0);
    const size_t* rid = (*row_set)[nid].begin;

    int64_t* hist_data = reinterpret_cast<int64_t*>((*histograms)[nid].Data());
    event_batch = qu->fill(hist_data, int64_t(0), 2 * nbins, event_batch);

    event_batch = qu->submit([&](::sycl::handler& cgh) {
      cgh.depends_on(event_batch);
      ::sycl::local_accessor<GradientPairInt64, 1> hist_fast(buffer_size, cgh);
      cgh.parallel_for<>(::sycl::nd_range<3>(::sycl::range<3>(nblocks, n_feature_groups, work_group_size),
                                             ::sycl::range<3>(      1,                1, work_group_size)),
                          [=](::sycl::nd_item<3> pid) {
        size_t row_block = pid.get_global_id(0);
        size_t feature_block = pid.get_global_id(1);
        size_t fid = foffsets_ptr[feature_block] + pid.get_global_id(2);
        if (fid < foffsets_ptr[feature_block + 1]) {
          size_t begin = row_block * block_size;
          size_t end = std::min(begin + block_size, n_rows);

          size_t first_bin = offsets[foffsets_ptr[feature_block]];
          for (int bin = offsets[fid] - first_bin; bin < offsets[fid + 1] - first_bin; bin += 1) {
            hist_fast[bin] = {0, 0};
          }

          for (size_t i = begin; i < end; ++i) {
            const size_t row_id = rid[i];
            const size_t icol_start = n_columns * row_id;
            const BinIdxType* gr_index_local = gradient_index + icol_start;

            uint32_t idx_bin = static_cast<uint32_t>(gr_index_local[fid]);
            idx_bin += offsets[fid] - first_bin;
            hist_fast[idx_bin] += pgh[row_id];
          }

          for (int bin = offsets[fid] - first_bin; bin < offsets[fid + 1] - first_bin; bin += 1) {
            AtomicRef<int64_t> grad(hist_data[2 * (bin + first_bin)]);
            AtomicRef<int64_t> hess(hist_data[2 * (bin + first_bin) + 1]);
            grad += hist_fast[bin].GetQuantisedGrad();
            hess += hist_fast[bin].GetQuantisedHess();
          }
        }
      });
    });
  }

  return event_batch;
}

static ::sycl::event BuildHistKernelL1(::sycl::queue* qu,
                              const GradientPairInt64* pgh,
                              const std::vector<bst_node_t>& nodes,
                              RowSetCollection* row_set,
                              const GHistIndexMatrix& gmat,
                              HistCollectionInt64* histograms,
                              GHistRowInt64* hist_buffer,
                              const DeviceProperties& device_prop,
                              ::sycl::event event) {
  using BinIdxType = uint8_t;
  constexpr int kMaxNumBins = 1u << (8 * sizeof(BinIdxType));
  const size_t n_columns = gmat.nfeatures;
  const BinIdxType* gradient_index = gmat.index.data<BinIdxType>();
  const uint32_t* offsets = gmat.cut.cut_ptrs_.ConstDevicePointer();
  const size_t nbins = gmat.nbins;
  size_t n_nodes = nodes.size();

  ::sycl::event event_batch = event;
  size_t max_wgs = std::min<size_t>(device_prop.max_sub_group_size * device_prop.eu_per_core,
                                    device_prop.max_work_group_size);
  size_t work_group_size = std::min<size_t>(n_columns, max_wgs);

  GradientPairInt64* hist_buffer_data = hist_buffer->Data();
  size_t nblocks = hist_buffer->Size() / nbins;

  for (size_t nidx = 0; nidx < n_nodes; ++nidx) {
    bst_node_t nid = nodes[nidx];
    size_t n_rows = (*row_set)[nid].Size();
    if (n_rows == 0) continue;

    size_t block_size = n_rows / nblocks + (n_rows % nblocks > 0);
    const size_t* rid = (*row_set)[nid].begin;

    event_batch = qu->submit([&](::sycl::handler& cgh) {
      cgh.depends_on(event_batch);
      cgh.parallel_for<>(::sycl::nd_range<2>(::sycl::range<2>(nblocks, work_group_size),
                                             ::sycl::range<2>(      1, work_group_size)),
                          [=](::sycl::nd_item<2> pid) {
        size_t block = pid.get_global_id(0);
        size_t feat = pid.get_global_id(1);

        GradientPairInt64* hist_local = hist_buffer_data + block * nbins;
        size_t begin = block * block_size;
        size_t end = std::min(begin + block_size, n_rows);

        GradientPairInt64 hist_fast[kMaxNumBins];
        for (size_t fid = feat; fid < n_columns; fid += work_group_size) {
          int n_bins_feature = offsets[fid+1] - offsets[fid];

          for (int bin = 0; bin < n_bins_feature; ++bin) {
            hist_fast[bin] = GradientPairInt64(0, 0);
          }

          for (size_t i = begin; i < end; ++i) {
            const size_t row_id = rid[i];
            const size_t icol_start = n_columns * row_id;
            const GradientPairInt64 pgh_row = pgh[row_id];
            const BinIdxType* gr_index_local = gradient_index + icol_start;
            BinIdxType bin = gr_index_local[fid];
            hist_fast[bin] += pgh_row;
          }

          for (int bin = 0; bin < n_bins_feature; ++bin) {
            hist_local[bin + offsets[fid]] = hist_fast[bin];
          }
        }
      });
    });

    GradientPairInt64* hist_data = (*histograms)[nid].Data();
    event_batch = ReduceHistParallel(qu, hist_data, hist_buffer_data, nblocks, nbins, event_batch);
  }

  return event_batch;
}

::sycl::event GHistBuilder::BuildHistL1(
              const GradientPairInt64* gpair_int64,
              const std::vector<bst_node_t>& nodes,
              RowSetCollection* row_indices,
              const GHistIndexMatrix& gmat,
              HistCollectionInt64* histograms,
              GHistRowInt64* hist_buffer,
              const DeviceProperties& device_prop,
              ::sycl::event event) {
  return BuildHistKernelL1(qu_, gpair_int64, nodes, row_indices, gmat, histograms, hist_buffer, device_prop, event);
}

}  // namespace common
}  // namespace sycl
}  // namespace xgboost
