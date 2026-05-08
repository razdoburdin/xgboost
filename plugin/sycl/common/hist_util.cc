/*!
 * Copyright 2017-2023 by Contributors
 * \file hist_util.cc
 */
#include <vector>
#include <limits>
#include <algorithm>

#include "../data/gradient_index.h"
#include "../tree/hist_dispatcher.h"
#include "hist_util.h"

#include <sycl/sycl.hpp>

namespace xgboost {
namespace sycl {
namespace common {

void CopyHist(::sycl::queue* qu,
              GHistRowInt64* dst,
              const GHistRowInt64& src,
              size_t size) {
  int64_t* pdst = reinterpret_cast<int64_t*>(dst->Data());
  const int64_t* psrc = reinterpret_cast<const int64_t*>(src.DataConst());

  qu->submit([&](::sycl::handler& cgh) {
    cgh.parallel_for<>(::sycl::range<1>(2 * size), [=](::sycl::item<1> pid) {
      const size_t i = pid.get_id(0);
      pdst[i] = psrc[i];
    });
  }).wait();
}

::sycl::event SubtractionHist(::sycl::queue* qu,
                              GHistRowInt64* dst,
                              const GHistRowInt64& src1,
                              const GHistRowInt64& src2,
                              size_t size, ::sycl::event event_priv) {
  int64_t* pdst = reinterpret_cast<int64_t*>(dst->Data());
  const int64_t* psrc1 = reinterpret_cast<const int64_t*>(src1.DataConst());
  const int64_t* psrc2 = reinterpret_cast<const int64_t*>(src2.DataConst());

  auto event_final = qu->submit([&](::sycl::handler& cgh) {
    cgh.depends_on(event_priv);
    cgh.parallel_for<>(::sycl::range<1>(2 * size), [pdst, psrc1, psrc2](::sycl::item<1> pid) {
      const size_t i = pid.get_id(0);
      pdst[i] = psrc1[i] - psrc2[i];
    });
  });
  return event_final;
}

// Single-node atomic kernel (used as fallback when hist doesn't fit in L1)
template<typename BinIdxType, bool isDense>
::sycl::event BuildHistKernel(::sycl::queue* qu,
                              const GradientPairInt64* pgh,
                              const RowSetCollection::Elem& row_indices,
                              const GHistIndexMatrix& gmat,
                              GHistRowInt64* hist,
                              ::sycl::event event_priv) {
  const size_t size = row_indices.Size();
  const size_t* rid = row_indices.begin;
  const size_t n_columns = isDense ? gmat.nfeatures : gmat.row_stride;
  const BinIdxType* gradient_index = gmat.index.data<BinIdxType>();
  const uint32_t* offsets = gmat.cut.cut_ptrs_.ConstDevicePointer();
  int64_t* hist_data = reinterpret_cast<int64_t*>(hist->Data());
  const size_t nbins = gmat.nbins;

  size_t work_group_size = std::min<size_t>(n_columns, 16);
  const size_t n_work_groups = n_columns / work_group_size + (n_columns % work_group_size > 0);

  auto event_main = qu->submit([&](::sycl::handler& cgh) {
    cgh.depends_on(event_priv);
    cgh.parallel_for<>(::sycl::nd_range<2>(::sycl::range<2>(size, n_work_groups * work_group_size),
                                           ::sycl::range<2>(1, work_group_size)),
                       [=](::sycl::nd_item<2> pid) {
      const int i = pid.get_global_id(0);
      auto group  = pid.get_group();

      const size_t icol_start = n_columns * rid[i];
      const size_t idx_gh = rid[i];
      const GradientPairInt64 gpair_q = pgh[idx_gh];
      const BinIdxType* gr_index_local = gradient_index + icol_start;

      const size_t group_id = group.get_group_id()[1];
      const size_t local_id = group.get_local_id()[1];
      const size_t j = group_id * work_group_size + local_id;
      if (j < n_columns) {
        uint32_t idx_bin = static_cast<uint32_t>(gr_index_local[j]);
        if constexpr (isDense) {
          idx_bin += offsets[j];
        }
        if (idx_bin < nbins) {
          AtomicRef<int64_t> gsum(hist_data[2 * idx_bin]);
          AtomicRef<int64_t> hsum(hist_data[2 * idx_bin + 1]);
          gsum += gpair_q.GetQuantisedGrad();
          hsum += gpair_q.GetQuantisedHess();
        }
      }
    });
  });
  return event_main;
}

// Buffer-based kernel: private histogram blocks per work-group, then atomic reduce
template<typename BinIdxType, bool isDense>
::sycl::event BuildHistKernel(::sycl::queue* qu,
                              const GradientPairInt64* pgh,
                              const std::vector<bst_node_t>& nodes,
                              const bst_node_t* nodes_ptr,
                              RowSetCollection* row_set,
                              const GHistIndexMatrix& gmat,
                              HistCollectionInt64* histograms,
                              GHistRowInt64* hist_buffer,
                              const DeviceProperties& device_prop,
                              ::sycl::event event) {
  const size_t n_columns = isDense ? gmat.nfeatures : gmat.row_stride;
  const BinIdxType* gradient_index = gmat.index.data<BinIdxType>();
  const uint32_t* offsets = gmat.cut.cut_ptrs_.ConstDevicePointer();
  const size_t nbins = gmat.nbins;
  size_t n_nodes = nodes.size();

  ::sycl::event event_batch = event;
  const auto* rows = row_set->RowSetDevice(qu, &event_batch);
  auto** hist_collection = histograms->GetDevicePointers();
  size_t work_group_size = std::min<size_t>(n_columns, device_prop.max_work_group_size);
  size_t n_wgs = n_columns / work_group_size + (n_columns % work_group_size > 0);

  GradientPairInt64* hist_buffer_data = hist_buffer->Data();
  size_t n_parallel_hist = hist_buffer->Size() / nbins;

  size_t n_row_blocks = n_parallel_hist;

  for (size_t nidx = 0; nidx < n_nodes; ++nidx) {
    bst_node_t nid = nodes[nidx];
    int64_t* hist = reinterpret_cast<int64_t*>((*histograms)[nid].Data());
    event_batch = qu->fill(hist, int64_t(0), 2 * nbins, event_batch);
  }

  event_batch = qu->submit([&](::sycl::handler& cgh) {
    cgh.depends_on(event_batch);
    cgh.parallel_for<>(::sycl::nd_range<2>(::sycl::range<2>(n_row_blocks, work_group_size),
                                           ::sycl::range<2>(           1, work_group_size)),
                        [=](::sycl::nd_item<2> pid) {
      const size_t hist_idx = pid.get_global_id(0);
      size_t feat = pid.get_global_id(1);

      const size_t row_block_idx = hist_idx;
      GradientPairInt64* hist_buff = hist_buffer_data + hist_idx * nbins;
      for (size_t nidx = 0; nidx < n_nodes; ++nidx) {
        bst_node_t nid = nodes_ptr[nidx];
        size_t n_rows = rows[nid].Size();
        if (n_rows > 0) {
          const size_t* rid = rows[nid].begin;
          int64_t* hist = reinterpret_cast<int64_t*>(hist_collection[nid]);
          for (size_t elem_idx = feat; elem_idx < nbins; elem_idx += work_group_size) {
            hist_buff[elem_idx] = GradientPairInt64(0, 0);
          }
          if constexpr (isDense) pid.barrier(::sycl::access::fence_space::local_space);

          for (size_t wg = 0; wg < n_wgs; ++wg) {
            size_t block_size = n_rows / n_row_blocks + (n_rows % n_row_blocks > 0);

            size_t begin = row_block_idx * block_size;
            size_t end = std::min(begin + block_size, n_rows);

            for (size_t i = begin; i < end; ++i) {
              const size_t icol_start = n_columns * rid[i];
              const size_t idx_gh = rid[i];
              const GradientPairInt64 gpair_q = pgh[idx_gh];
              const BinIdxType* gr_index_local = gradient_index + icol_start;

              size_t fid = feat + work_group_size * wg;
              if constexpr (!isDense) pid.barrier(::sycl::access::fence_space::local_space);
              if (fid < n_columns) {
                uint32_t idx_bin = static_cast<uint32_t>(gr_index_local[fid]);
                if constexpr (isDense) {
                  idx_bin += offsets[fid];
                }
                if (idx_bin < nbins) {
                  hist_buff[idx_bin] += gpair_q;
                }
              }
            }
          }

          pid.barrier(::sycl::access::fence_space::local_space);
          for (size_t elem_idx = feat; elem_idx < nbins; elem_idx += work_group_size) {
            AtomicRef<int64_t> gsum(hist[2 * elem_idx]);
            AtomicRef<int64_t> hsum(hist[2 * elem_idx + 1]);
            gsum += hist_buff[elem_idx].GetQuantisedGrad();
            hsum += hist_buff[elem_idx].GetQuantisedHess();
          }
        }
      }
    });
  });

  return event_batch;
}

// Batch-atomic kernel: direct atomic accumulation, L2-batched across nodes.
// Falls back to single-node kernel when histogram doesn't fit in L1.
template<typename BinIdxType, bool isDense>
::sycl::event BuildHistKernel(::sycl::queue* qu,
                              const GradientPairInt64* pgh,
                              const std::vector<bst_node_t>& nodes,
                              const bst_node_t* nodes_ptr,
                              RowSetCollection* row_set,
                              const GHistIndexMatrix& gmat,
                              HistCollectionInt64* histograms,
                              const DeviceProperties& device_prop,
                              ::sycl::event event) {
  const size_t n_columns = isDense ? gmat.nfeatures : gmat.row_stride;
  size_t work_group_size = std::min<size_t>(n_columns, 16);
  size_t n_nodes = nodes.size();

  const size_t nbins = gmat.nbins;
  size_t l2_size = device_prop.l2_size;
  size_t l1_size = device_prop.l1_size;
  size_t hist_size = 2 * sizeof(int64_t) * nbins;

  // Fallback: when histogram doesn't fit in L1, build each node separately
  if (hist_size > l1_size) {
    std::vector<::sycl::event> events(n_nodes);
    for (size_t nidx = 0; nidx < n_nodes; ++nidx) {
      bst_node_t nid = nodes[nidx];
      int64_t* hist = reinterpret_cast<int64_t*>((*histograms)[nid].Data());
      events[nidx] = qu->fill(hist, int64_t(0), 2 * nbins, event);
      events[nidx] = BuildHistKernel<BinIdxType, isDense>(
          qu, pgh, (*row_set)[nid], gmat, &((*histograms)[nid]), events[nidx]);
    }
    return qu->submit([&](::sycl::handler& cgh) {
        cgh.depends_on(events);
    });
  }

  const BinIdxType* gradient_index = gmat.index.data<BinIdxType>();
  const uint32_t* offsets = gmat.cut.cut_ptrs_.ConstDevicePointer();

  const size_t n_work_groups = n_columns / work_group_size + (n_columns % work_group_size > 0);

  ::sycl::event event_batch = event;
  const auto* rows = row_set->RowSetDevice(qu, &event_batch);
  auto** hist_collection = histograms->GetDevicePointers();

  size_t node_batch_size = l2_size / hist_size;
  if (node_batch_size > n_nodes) node_batch_size = n_nodes;
  if (node_batch_size == 0) node_batch_size = 1;
  size_t n_node_batch = n_nodes / node_batch_size + (n_nodes % node_batch_size > 0);

  std::vector<::sycl::event> events(node_batch_size);
  for (size_t node_batch = 0; node_batch < n_node_batch; ++node_batch) {
    size_t first_node = node_batch * node_batch_size;
    size_t last_node = std::min<size_t>(first_node + node_batch_size, n_nodes);
    size_t nodes_in_batch = last_node - first_node;

    size_t max_size = 0;
    for (size_t nidx = 0; nidx < nodes_in_batch; ++nidx) {
      bst_node_t nid = nodes[nidx + first_node];
      max_size = std::max(max_size, (*row_set)[nid].Size());
      int64_t* hist = reinterpret_cast<int64_t*>((*histograms)[nid].Data());
      events[nidx] = qu->fill(hist, int64_t(0), 2 * nbins, event_batch);
    }

    size_t max_block_size = 32;
    size_t n_blocks = max_size / max_block_size + (max_size % max_block_size > 0);

    size_t n_sub_groups = work_group_size / device_prop.min_sub_group_size
                        + (work_group_size % device_prop.min_sub_group_size > 0);
    size_t n_sub_groups_per_core = std::min<size_t>(device_prop.eu_per_core, n_sub_groups);

    constexpr float kMaxGPUUtilisation = 4;
    n_blocks = std::max<size_t>(n_blocks, kMaxGPUUtilisation * device_prop.max_compute_units / (n_sub_groups_per_core * nodes_in_batch * n_work_groups));

    event_batch = qu->submit([&](::sycl::handler& cgh) {
      cgh.depends_on(events);
      cgh.parallel_for<>(::sycl::nd_range<3>(::sycl::range<3>(n_blocks, nodes_in_batch, n_work_groups * work_group_size),
                                              ::sycl::range<3>(       1,              1, work_group_size)),
                        [=](::sycl::nd_item<3> pid) {
        auto group  = pid.get_group();
        const size_t block    = pid.get_global_id(0);
        const size_t node_idx = pid.get_global_id(1) + first_node;
        const size_t group_id = group.get_group_id()[2];
        const size_t local_id = group.get_local_id()[2];
        const size_t j = group_id * work_group_size + local_id;

        bst_node_t nid = nodes_ptr[node_idx];
        size_t n_rows = rows[nid].Size();
        if ((j < n_columns) && (n_rows > 0)) {
          const size_t* rid = rows[nid].begin;
          int64_t* hist = reinterpret_cast<int64_t*>(hist_collection[nid]);

          size_t block_size = n_rows / n_blocks + (n_rows % n_blocks > 0);

          size_t begin = block * block_size;
          size_t end = std::min(begin + block_size, n_rows);

          for (size_t i = begin; i < end; ++i) {
            const size_t icol_start = n_columns * rid[i];
            const size_t idx_gh = rid[i];
            const GradientPairInt64 gpair_q = pgh[idx_gh];
            const BinIdxType* gr_index_local = gradient_index + icol_start;

            uint32_t idx_bin = static_cast<uint32_t>(gr_index_local[j]);
            if constexpr (isDense) {
              idx_bin += offsets[j];
            }

            AtomicRef<int64_t> gsum(hist[2 * idx_bin]);
            AtomicRef<int64_t> hsum(hist[2 * idx_bin + 1]);
            gsum += gpair_q.GetQuantisedGrad();
            hsum += gpair_q.GetQuantisedHess();
          }
        }
      });
    });
  }

  return event_batch;
}

// Dispatcher: buffer-based BuildHist
::sycl::event GHistBuilder::BuildHist(
              const GradientPairInt64* gpair_int64,
              const std::vector<bst_node_t>& nodes,
              const bst_node_t* nodes_device_ptr,
              RowSetCollection* row_indices,
              const GHistIndexMatrix& gmat,
              HistCollectionInt64* histograms,
              GHistRowInt64* hist_buffer,
              const DeviceProperties& device_prop,
              ::sycl::event event) {
  switch (gmat.index.GetBinTypeSize()) {
    case BinTypeSize::kUint8BinsTypeSize:
      if (gmat.IsDense()) {
        return BuildHistKernel<uint8_t, true>(qu_, gpair_int64, nodes, nodes_device_ptr, row_indices, gmat, histograms, hist_buffer, device_prop, event);
      } else {
        return BuildHistKernel<uint32_t, false>(qu_, gpair_int64, nodes, nodes_device_ptr, row_indices, gmat, histograms, hist_buffer, device_prop, event);
      }
      break;
    case BinTypeSize::kUint16BinsTypeSize:
      if (gmat.IsDense()) {
        return BuildHistKernel<uint16_t, true>(qu_, gpair_int64, nodes, nodes_device_ptr, row_indices, gmat, histograms, hist_buffer, device_prop, event);
      } else {
        return BuildHistKernel<uint32_t, false>(qu_, gpair_int64, nodes, nodes_device_ptr, row_indices, gmat, histograms, hist_buffer, device_prop, event);
      }
      break;
    case BinTypeSize::kUint32BinsTypeSize:
      if (gmat.IsDense()) {
        return BuildHistKernel<uint32_t, true>(qu_, gpair_int64, nodes, nodes_device_ptr, row_indices, gmat, histograms, hist_buffer, device_prop, event);
      } else {
        return BuildHistKernel<uint32_t, false>(qu_, gpair_int64, nodes, nodes_device_ptr, row_indices, gmat, histograms, hist_buffer, device_prop, event);
      }
      break;
    default:
      CHECK(false);  // no default behavior
  }
}

// Dispatcher: atomic-based BuildHist (with single-node fallback)
::sycl::event GHistBuilder::BuildHist(
              const GradientPairInt64* gpair_int64,
              const std::vector<bst_node_t>& nodes,
              const bst_node_t* nodes_ptr,
              RowSetCollection* row_indices,
              const GHistIndexMatrix& gmat,
              HistCollectionInt64* histograms,
              const DeviceProperties& device_prop,
              ::sycl::event event) {
  switch (gmat.index.GetBinTypeSize()) {
    case BinTypeSize::kUint8BinsTypeSize:
      if (gmat.IsDense()) {
        return BuildHistKernel<uint8_t, true>(qu_, gpair_int64, nodes, nodes_ptr, row_indices, gmat, histograms, device_prop, event);
      } else {
        return BuildHistKernel<uint32_t, false>(qu_, gpair_int64, nodes, nodes_ptr, row_indices, gmat, histograms, device_prop, event);
      }
      break;
    case BinTypeSize::kUint16BinsTypeSize:
      if (gmat.IsDense()) {
        return BuildHistKernel<uint16_t, true>(qu_, gpair_int64, nodes, nodes_ptr, row_indices, gmat, histograms, device_prop, event);
      } else {
        return BuildHistKernel<uint32_t, false>(qu_, gpair_int64, nodes, nodes_ptr, row_indices, gmat, histograms, device_prop, event);
      }
      break;
    case BinTypeSize::kUint32BinsTypeSize:
      if (gmat.IsDense()) {
        return BuildHistKernel<uint32_t, true>(qu_, gpair_int64, nodes, nodes_ptr, row_indices, gmat, histograms, device_prop, event);
      } else {
        return BuildHistKernel<uint32_t, false>(qu_, gpair_int64, nodes, nodes_ptr, row_indices, gmat, histograms, device_prop, event);
      }
      break;
    default:
      CHECK(false);  // no default behavior
  }
}

}  // namespace common
}  // namespace sycl
}  // namespace xgboost
