/*!
 * Copyright 2017-2023 by Contributors
 * \file hist_util.h
 */
#ifndef PLUGIN_SYCL_COMMON_HIST_UTIL_H_
#define PLUGIN_SYCL_COMMON_HIST_UTIL_H_

#include <vector>
#include <unordered_map>
#include <memory>

#include "../data.h"
#include "row_set.h"

#include "../../src/common/hist_util.h"
#include "../data/gradient_index.h"
#include "../tree/hist_dispatcher.h"
#include "../tree/gradient_quantiser.h"

#include <sycl/sycl.hpp>

namespace xgboost {
namespace sycl {
namespace common {

using GHistRowInt64 = USMVector<GradientPairInt64, MemoryType::on_device>;

using BinTypeSize = ::xgboost::common::BinTypeSize;

class ColumnMatrix;

void CopyHist(::sycl::queue* qu,
              GHistRowInt64* dst,
              const GHistRowInt64& src,
              size_t size);

::sycl::event SubtractionHist(::sycl::queue* qu,
                              GHistRowInt64* dst,
                              const GHistRowInt64& src1,
                              const GHistRowInt64& src2,
                              size_t size, ::sycl::event event_priv);

/*!
 * \brief Int64 quantized histograms of gradient statistics for multiple nodes
 */
class HistCollectionInt64 {
 public:
  GHistRowInt64& operator[](bst_uint nid) {
    return *(data_.at(nid));
  }

  const GHistRowInt64& operator[](bst_uint nid) const {
    return *(data_.at(nid));
  }

  void Init(::sycl::queue* qu, uint32_t nbins) {
    qu_ = qu;
    if (nbins_ != nbins) {
      nbins_ = nbins;
      data_.clear();
    }
  }

  void AddHistRow(bst_uint nid) {
    if (data_.count(nid) == 0) {
      data_[nid] = std::make_unique<GHistRowInt64>(qu_, nbins_);
    }
  }

  void PushPointersToDevice() {
    std::vector<GradientPairInt64*> ptrs_host(data_.size(), nullptr);
    for (const auto& [nid, hist] : data_) {
      if (ptrs_host.size() <= nid) ptrs_host.resize(nid + 1);
      ptrs_host[nid] = hist->Data();
    }
    ptrs_.Init(qu_, ptrs_host);
  }

  GradientPairInt64** GetDevicePointers() {
    return ptrs_.Data();
  }

 private:
  uint32_t nbins_ = 0;
  std::unordered_map<uint32_t, std::unique_ptr<GHistRowInt64>> data_;
  USMVector<GradientPairInt64*, MemoryType::on_device> ptrs_;
  ::sycl::queue* qu_;
};

/*!
 * \brief Int64 temporary histograms for parallel computation
 */
class ParallelGHistBuilderInt64 {
 public:
  void Init(::sycl::queue* qu, size_t nbins) {
    qu_ = qu;
    if (nbins != nbins_) {
      nbins_ = nbins;
    }
  }

  void Reset(size_t nblocks) {
    hist_device_buffer_.Resize(qu_, nblocks * nbins_);
  }

  GHistRowInt64& GetDeviceBuffer() {
    return hist_device_buffer_;
  }

 private:
  size_t nbins_ = 0;
  GHistRowInt64 hist_device_buffer_;
  ::sycl::queue* qu_;
};

/*!
 * \brief Builder for int64 quantized histograms of gradient statistics
 */
class GHistBuilder {
 public:
  GHistBuilder() = default;
  GHistBuilder(::sycl::queue* qu, uint32_t nbins) : qu_{qu}, nbins_{nbins} {}

  // Buffer-based: accumulate in private hist blocks then atomically reduce
  ::sycl::event BuildHist(const GradientPairInt64* gpair_int64,
                          const std::vector<bst_node_t>& nodes,
                          const bst_node_t* nodes_device_ptr,
                          RowSetCollection* row_indices,
                          const GHistIndexMatrix& gmat,
                          HistCollectionInt64* histograms,
                          GHistRowInt64* hist_buffer,
                          const DeviceProperties& device_prop,
                          ::sycl::event event);

  // Atomic-based: direct atomic accumulation, L2-batched across nodes
  ::sycl::event BuildHist(const GradientPairInt64* gpair_int64,
                          const std::vector<bst_node_t>& nodes,
                          const bst_node_t* nodes_device_ptr,
                          RowSetCollection* row_indices,
                          const GHistIndexMatrix& gmat,
                          HistCollectionInt64* histograms,
                          const DeviceProperties& device_prop,
                          ::sycl::event event);

  // L1-optimized: per-feature accumulation in registers then reduce
  ::sycl::event BuildHistL1(const GradientPairInt64* gpair_int64,
                            const std::vector<bst_node_t>& nodes,
                            RowSetCollection* row_indices,
                            const GHistIndexMatrix& gmat,
                            HistCollectionInt64* histograms,
                            GHistRowInt64* hist_buffer,
                            const DeviceProperties& device_prop,
                            ::sycl::event event);

  uint32_t GetNumBins() const {
      return nbins_;
  }

 private:
  uint32_t nbins_ { 0 };
  ::sycl::queue* qu_;
};
}  // namespace common
}  // namespace sycl
}  // namespace xgboost
#endif  // PLUGIN_SYCL_COMMON_HIST_UTIL_H_
