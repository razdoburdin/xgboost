/*!
 * Copyright 2017-2023 by Contributors
 * \file hist_util.cc
 */
#include <vector>
#include <limits>
#include <algorithm>

#include "../data/gradient_index.h"
#include "hist_util.h"

#include <sycl/sycl.hpp>

namespace xgboost {
namespace sycl {
namespace common {

/*!
 * \brief Fill histogram with zeroes
 */
template<typename GradientSumT>
::sycl::event InitHist(::sycl::queue* qu,
                       GHistRow<GradientSumT, MemoryType::on_device>* hist,
                       size_t size, ::sycl::event event_in) {
  return qu->fill(hist->Begin(),
                  xgboost::detail::GradientPairInternal<GradientSumT>(), size, event_in);
}
template ::sycl::event InitHist(::sycl::queue* qu,
                                GHistRow<float,  MemoryType::on_device>* hist,
                                size_t size, ::sycl::event event_in);
template ::sycl::event InitHist(::sycl::queue* qu,
                                GHistRow<double, MemoryType::on_device>* hist,
                                size_t size, ::sycl::event event_in);

/*!
 * \brief Copy histogram from src to dst
 */
template<typename GradientSumT>
void CopyHist(::sycl::queue* qu,
              GHistRow<GradientSumT, MemoryType::on_device>* dst,
              const GHistRow<GradientSumT, MemoryType::on_device>& src,
              size_t size) {
  GradientSumT* pdst = reinterpret_cast<GradientSumT*>(dst->Data());
  const GradientSumT* psrc = reinterpret_cast<const GradientSumT*>(src.DataConst());

  qu->submit([&](::sycl::handler& cgh) {
    cgh.parallel_for<>(::sycl::range<1>(2 * size), [=](::sycl::item<1> pid) {
      const size_t i = pid.get_id(0);
      pdst[i] = psrc[i];
    });
  }).wait();
}
template void CopyHist(::sycl::queue* qu,
                       GHistRow<float, MemoryType::on_device>* dst,
                       const GHistRow<float, MemoryType::on_device>& src,
                       size_t size);
template void CopyHist(::sycl::queue* qu,
                       GHistRow<double, MemoryType::on_device>* dst,
                       const GHistRow<double, MemoryType::on_device>& src,
                       size_t size);

/*!
 * \brief Compute Subtraction: dst = src1 - src2
 */
template<typename GradientSumT>
::sycl::event SubtractionHist(::sycl::queue* qu,
                            GHistRow<GradientSumT, MemoryType::on_device>* dst,
                            const GHistRow<GradientSumT, MemoryType::on_device>& src1,
                            const GHistRow<GradientSumT, MemoryType::on_device>& src2,
                            size_t size, const std::vector<::sycl::event>& events_in) {
  GradientSumT* pdst = reinterpret_cast<GradientSumT*>(dst->Data());
  const GradientSumT* psrc1 = reinterpret_cast<const GradientSumT*>(src1.DataConst());
  const GradientSumT* psrc2 = reinterpret_cast<const GradientSumT*>(src2.DataConst());

  auto event_final = qu->submit([&](::sycl::handler& cgh) {
    cgh.depends_on(events_in);
    cgh.parallel_for<>(::sycl::range<1>(2 * size), [pdst, psrc1, psrc2](::sycl::item<1> pid) {
      const size_t i = pid.get_id(0);
      pdst[i] = psrc1[i] - psrc2[i];
    });
  });
  return event_final;
}
template ::sycl::event SubtractionHist(::sycl::queue* qu,
                              GHistRow<float, MemoryType::on_device>* dst,
                              const GHistRow<float, MemoryType::on_device>& src1,
                              const GHistRow<float, MemoryType::on_device>& src2,
                              size_t size, const std::vector<::sycl::event>& events_in);
template ::sycl::event SubtractionHist(::sycl::queue* qu,
                              GHistRow<double, MemoryType::on_device>* dst,
                              const GHistRow<double, MemoryType::on_device>& src1,
                              const GHistRow<double, MemoryType::on_device>& src2,
                              size_t size, const std::vector<::sycl::event>& events_in);

inline auto GetBlocksParameters(::sycl::queue* qu, size_t size, size_t max_nblocks) {
  struct _ {
    size_t block_size, nblocks;
  };

  const size_t min_block_size = 32;
  const size_t max_compute_units =
    qu->get_device().get_info<::sycl::info::device::max_compute_units>();

  size_t nblocks = max_compute_units;

  size_t block_size = size / nblocks + !!(size % nblocks);
  if (block_size > (1u << 12)) {
    nblocks = max_nblocks;
    block_size = size / nblocks + !!(size % nblocks);
  }
  if (block_size < min_block_size) {
    block_size = min_block_size;
    nblocks = size / block_size + !!(size % block_size);
  }

  return _{block_size, nblocks};
}

// Kernel with buffer using
template<typename FPType, typename BinIdxType, bool isDense>
::sycl::event BuildHistKernel(::sycl::queue* qu,
                            const HostDeviceVector<GradientPair>& gpair,
                            const RowSetCollection::Elem& row_indices,
                            const GHistIndexMatrix& gmat,
                            GHistRow<FPType, MemoryType::on_device>* hist,
                            GHistRow<FPType, MemoryType::on_device>* hist_buffer,
                            const std::vector<::sycl::event>& events_in) {
  using GradientPairT = xgboost::detail::GradientPairInternal<FPType>;
  const size_t size = row_indices.Size();
  const size_t* rid = row_indices.begin;
  const size_t n_columns = isDense ? gmat.nfeatures : gmat.row_stride;
  const auto* pgh = gpair.ConstDevicePointer();
  const BinIdxType* gradient_index = gmat.index.data<BinIdxType>();
  const uint32_t* offsets = gmat.cut.cut_ptrs_.ConstDevicePointer();
  const size_t nbins = gmat.nbins;

  const size_t max_work_group_size =
    qu->get_device().get_info<::sycl::info::device::max_work_group_size>();
  const size_t work_group_size = n_columns < max_work_group_size ? n_columns : max_work_group_size;

  // Captured structured bindings are a C++20 extension
  const auto block_params = GetBlocksParameters(qu, size, hist_buffer->Size() / (nbins * 2));
  const size_t block_size = block_params.block_size;
  const size_t nblocks = block_params.nblocks;

  GradientPairT* hist_buffer_data = hist_buffer->Data();
  auto event_fill = qu->fill(hist_buffer_data, GradientPairT(0, 0),
                             nblocks * nbins * 2, events_in);
  auto event_main = qu->submit([&](::sycl::handler& cgh) {
    cgh.depends_on(event_fill);
    cgh.parallel_for<>(::sycl::nd_range<2>(::sycl::range<2>(nblocks, work_group_size),
                                           ::sycl::range<2>(1, work_group_size)),
                       [=](::sycl::nd_item<2> pid) {
      size_t block = pid.get_global_id(0);
      size_t feat = pid.get_global_id(1);

      GradientPairT* hist_local = hist_buffer_data + block * nbins;
      for (size_t idx = 0; idx < block_size; ++idx) {
        size_t i = block * block_size + idx;
        if (i < size) {
          const size_t icol_start = n_columns * rid[i];
          const size_t idx_gh = rid[i];

          const GradientPairT pgh_row = {pgh[idx_gh].GetGrad(), pgh[idx_gh].GetHess()};
          pid.barrier(::sycl::access::fence_space::local_space);
          const BinIdxType* gr_index_local = gradient_index + icol_start;

          for (size_t j = feat; j < n_columns; j += work_group_size) {
            uint32_t idx_bin = static_cast<uint32_t>(gr_index_local[j]);
            if constexpr (isDense) {
              idx_bin += offsets[j];
            }
            if (idx_bin < nbins) {
              hist_local[idx_bin] += pgh_row;
            }
          }
        }
      }
    });
  });

  GradientPairT* hist_data = hist->Data();
  auto event_save = qu->submit([&](::sycl::handler& cgh) {
    cgh.depends_on(event_main);
    cgh.parallel_for<>(::sycl::range<1>(nbins), [=](::sycl::item<1> pid) {
      size_t idx_bin = pid.get_id(0);

      GradientPairT gpair = {0, 0};

      for (size_t j = 0; j < nblocks; ++j) {
        gpair += hist_buffer_data[j * nbins + idx_bin];
      }

      hist_data[idx_bin] = gpair;
    });
  });
  return event_save;
}

// Kernel with atomic using
template<typename FPType, typename BinIdxType, bool isDense>
::sycl::event BuildHistKernel(::sycl::queue* qu,
                            const HostDeviceVector<GradientPair>& gpair,
                            const RowSetCollection::Elem& row_indices,
                            const GHistIndexMatrix& gmat,
                            GHistRow<FPType, MemoryType::on_device>* hist,
                            ::sycl::event event_in) {
  const size_t size = row_indices.Size();
  const size_t* rid = row_indices.begin;
  const size_t n_columns = isDense ? gmat.nfeatures : gmat.row_stride;
  const GradientPair::ValueT* pgh =
    reinterpret_cast<const GradientPair::ValueT*>(gpair.ConstDevicePointer());
  const BinIdxType* gradient_index = gmat.index.data<BinIdxType>();
  const uint32_t* offsets = gmat.cut.cut_ptrs_.ConstDevicePointer();
  FPType* hist_data = reinterpret_cast<FPType*>(hist->Data());
  const size_t nbins = gmat.nbins;

  constexpr size_t work_group_size = 32;
  const size_t n_work_groups = n_columns / work_group_size + (n_columns % work_group_size > 0);

  auto event_fill = qu->fill(hist_data, FPType(0), nbins * 2, event_in);
  auto event_main = qu->submit([&](::sycl::handler& cgh) {
    cgh.depends_on(event_fill);
    cgh.parallel_for<>(::sycl::nd_range<2>(::sycl::range<2>(size, n_work_groups * work_group_size),
                                           ::sycl::range<2>(1, work_group_size)),
                       [=](::sycl::nd_item<2> pid) {
      const int i = pid.get_global_id(0);
      auto group  = pid.get_group();

      const size_t icol_start = n_columns * rid[i];
      const size_t idx_gh = rid[i];
      const FPType pgh_row[2] = {pgh[2 * idx_gh], pgh[2 * idx_gh + 1]};
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
          AtomicRef<FPType> gsum(hist_data[2 * idx_bin]);
          AtomicRef<FPType> hsum(hist_data[2 * idx_bin + 1]);
          gsum += pgh_row[0];
          hsum += pgh_row[1];
        }
      }
    });
  });
  return event_main;
}

// Kernel with atomic using
template<typename FPType, typename BinIdxType, bool isDense>
::sycl::event BuildHistKernel0(::sycl::queue* qu,
                            const HostDeviceVector<GradientPair>& gpair,
                            const RowSetCollection::Elem& row_indices,
                            const GHistIndexMatrix& gmat,
                            GHistRow<FPType, MemoryType::on_device>* hist,
                            ::sycl::event event_in) {
  const size_t size = row_indices.Size();
  const size_t* rid = row_indices.begin;
  const size_t n_columns = isDense ? gmat.nfeatures : gmat.row_stride;
  const GradientPair::ValueT* pgh =
    reinterpret_cast<const GradientPair::ValueT*>(gpair.ConstDevicePointer());
  const BinIdxType* gradient_index = gmat.index.data<BinIdxType>();
  const uint32_t* offsets = gmat.cut.cut_ptrs_.ConstDevicePointer();
  FPType* hist_data = reinterpret_cast<FPType*>(hist->Data());
  const size_t nbins = gmat.nbins;

  constexpr size_t work_group_size = 32;
  const size_t n_work_groups = (size + work_group_size - 1) / work_group_size;
  const size_t global_size = n_work_groups * work_group_size;

  auto event_fill = qu->fill(hist_data, FPType(0), nbins * 2, event_in);
  auto event_main = qu->submit([&](::sycl::handler& cgh) {
    cgh.depends_on(event_fill);

    ::sycl::local_accessor<FPType, 1> local_hist(::sycl::range<1>(nbins * 2), cgh);
    cgh.parallel_for<>(::sycl::nd_range<1>(global_size, work_group_size),
                       [=](::sycl::nd_item<1> item) {
      const int local_id = item.get_local_id(0);
      const int group_size = item.get_local_range(0);

      for (size_t i = local_id; i < nbins * 2; i += group_size) {
        local_hist[i] = 0.0;
      }
      item.barrier(::sycl::access::fence_space::local_space);

      for (size_t i = item.get_global_id(0); i < size; i += item.get_global_range(0)) {
        const size_t icol_start = n_columns * rid[i];
        const size_t idx_gh = rid[i];
        const FPType pgh_row[2] = {pgh[2 * idx_gh], pgh[2 * idx_gh + 1]};
        const BinIdxType* gr_index_local = gradient_index + icol_start;
        for (size_t j = 0; j < n_columns; ++j) {
          uint32_t idx_bin = static_cast<uint32_t>(gr_index_local[j]);
          if constexpr (isDense) {
            idx_bin += offsets[j];
          }
          if (idx_bin < nbins) {
            LocalAtomicRef<FPType> gsum(local_hist[2 * idx_bin]);
            LocalAtomicRef<FPType> hsum(local_hist[2 * idx_bin+1]);
            gsum += pgh_row[0];
            hsum += pgh_row[1];
          }
        }
      }

      item.barrier(::sycl::access::fence_space::local_space);

      for (size_t i = local_id; i < nbins; i += group_size) {
        AtomicRef<FPType> gsum(hist_data[2 * i]);
        AtomicRef<FPType> hsum(hist_data[2 * i + 1]);
        gsum += local_hist[2 * i];
        hsum += local_hist[2 * i + 1];
      }
    });
  });
  return event_main;
}

template<typename FPType, typename BinIdxType>
::sycl::event BuildHistDispatchKernel(
                ::sycl::queue* qu,
                const HostDeviceVector<GradientPair>& gpair,
                const RowSetCollection::Elem& row_indices,
                const GHistIndexMatrix& gmat,
                GHistRow<FPType, MemoryType::on_device>* hist,
                bool isDense,
                GHistRow<FPType, MemoryType::on_device>* hist_buffer,
                ::sycl::event event_in,
                std::vector<::sycl::event>* events_buffer,
                bool force_atomic_use) {
  const size_t size = row_indices.Size();
  const size_t n_columns = isDense ? gmat.nfeatures : gmat.row_stride;
  const size_t nbins = gmat.nbins;

  // TODO(razdoburdin): replace the add-hock dispatching criteria by more sutable one
  bool use_atomic = (size < nbins) || (gmat.max_num_bins == gmat.nbins / n_columns);

  // force_atomic_use flag is used only for testing
  use_atomic = use_atomic || force_atomic_use;
  if (!use_atomic) {
    ::sycl::event event_out;
    events_buffer->push_back(event_in);
    if (isDense) {
      event_out = BuildHistKernel<FPType, BinIdxType, true>(qu, gpair, row_indices,
                                                            gmat, hist, hist_buffer,
                                                            *events_buffer);
    } else {
      event_out = BuildHistKernel<FPType, uint32_t, false>(qu, gpair, row_indices,
                                                           gmat, hist, hist_buffer,
                                                           *events_buffer);
    }
    events_buffer->back() = event_out;
    return event_out;
  } else {
    if (isDense) {
      return BuildHistKernel<FPType, BinIdxType, true>(qu, gpair, row_indices,
                                                       gmat, hist, event_in);
    } else {
      return BuildHistKernel<FPType, uint32_t, false>(qu, gpair, row_indices,
                                                      gmat, hist, event_in);
    }
  }
}

template<typename FPType>
::sycl::event BuildHistKernel(::sycl::queue* qu,
                            const HostDeviceVector<GradientPair>& gpair,
                            const RowSetCollection::Elem& row_indices,
                            const GHistIndexMatrix& gmat, const bool isDense,
                            GHistRow<FPType, MemoryType::on_device>* hist,
                            GHistRow<FPType, MemoryType::on_device>* hist_buffer,
                            ::sycl::event event_in,
                            std::vector<::sycl::event>* events_buffer,
                            bool force_atomic_use) {
  const bool is_dense = isDense;
  switch (gmat.index.GetBinTypeSize()) {
    case BinTypeSize::kUint8BinsTypeSize:
      return BuildHistDispatchKernel<FPType, uint8_t>(qu, gpair, row_indices,
                                                      gmat, hist, is_dense, hist_buffer,
                                                      event_in, events_buffer, force_atomic_use);
      break;
    case BinTypeSize::kUint16BinsTypeSize:
      return BuildHistDispatchKernel<FPType, uint16_t>(qu, gpair, row_indices,
                                                       gmat, hist, is_dense, hist_buffer,
                                                       event_in, events_buffer, force_atomic_use);
      break;
    case BinTypeSize::kUint32BinsTypeSize:
      return BuildHistDispatchKernel<FPType, uint32_t>(qu, gpair, row_indices,
                                                       gmat, hist, is_dense, hist_buffer,
                                                       event_in, events_buffer, force_atomic_use);
      break;
    default:
      CHECK(false);  // no default behavior
  }
}

template <typename GradientSumT>
::sycl::event GHistBuilder<GradientSumT>::BuildHist(
              const HostDeviceVector<GradientPair>& gpair,
              const RowSetCollection::Elem& row_indices,
              const GHistIndexMatrix &gmat,
              GHistRowT<MemoryType::on_device>* hist,
              bool isDense,
              GHistRowT<MemoryType::on_device>* hist_buffer,
              ::sycl::event event_in,
              std::vector<::sycl::event>* events_buffer,
              bool force_atomic_use) {
  return BuildHistKernel<GradientSumT>(qu_, gpair, row_indices, gmat,
                                       isDense, hist, hist_buffer, event_in, events_buffer,
                                       force_atomic_use);
}

template
::sycl::event GHistBuilder<float>::BuildHist(
              const HostDeviceVector<GradientPair>& gpair,
              const RowSetCollection::Elem& row_indices,
              const GHistIndexMatrix& gmat,
              GHistRow<float, MemoryType::on_device>* hist,
              bool isDense,
              GHistRow<float, MemoryType::on_device>* hist_buffer,
              ::sycl::event event_in,
              std::vector<::sycl::event>* events_buffer,
              bool force_atomic_use);
template
::sycl::event GHistBuilder<double>::BuildHist(
              const HostDeviceVector<GradientPair>& gpair,
              const RowSetCollection::Elem& row_indices,
              const GHistIndexMatrix& gmat,
              GHistRow<double, MemoryType::on_device>* hist,
              bool isDense,
              GHistRow<double, MemoryType::on_device>* hist_buffer,
              ::sycl::event event_in,
              std::vector<::sycl::event>* events_buffer,
              bool force_atomic_use);

}  // namespace common
}  // namespace sycl
}  // namespace xgboost
