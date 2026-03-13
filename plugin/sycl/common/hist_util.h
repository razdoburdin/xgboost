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

#include <sycl/sycl.hpp>

namespace xgboost {
namespace sycl {
namespace common {

template<typename GradientSumT, MemoryType memory_type = MemoryType::shared>
using GHistRow = USMVector<xgboost::detail::GradientPairInternal<GradientSumT>, memory_type>;

using BinTypeSize = ::xgboost::common::BinTypeSize;

class ColumnMatrix;

/*!
 * \brief Copy histogram from src to dst
 */
template<typename GradientSumT>
void CopyHist(::sycl::queue* qu,
              GHistRow<GradientSumT, MemoryType::on_device>* dst,
              const GHistRow<GradientSumT, MemoryType::on_device>& src,
              size_t size);

/*!
 * \brief Compute subtraction: dst = src1 - src2
 */
template<typename GradientSumT>
::sycl::event SubtractionHist(::sycl::queue* qu,
                              GHistRow<GradientSumT, MemoryType::on_device>* dst,
                              const GHistRow<GradientSumT, MemoryType::on_device>& src1,
                              const GHistRow<GradientSumT, MemoryType::on_device>& src2,
                              size_t size, ::sycl::event event_priv);

/*!
 * \brief Histograms of gradient statistics for multiple nodes
 */
template<typename GradientSumT>
class HistCollection {
 public:
  using GHistRowT = GHistRow<GradientSumT, MemoryType::on_device>;
  using GradientPair = xgboost::detail::GradientPairInternal<GradientSumT>;

  // Access histogram for i-th node
  GHistRowT& operator[](bst_uint nid) {
    return *(data_.at(nid));
  }

  const GHistRowT& operator[](bst_uint nid) const {
    return *(data_.at(nid));
  }

  // Initialize histogram collection
  void Init(::sycl::queue* qu, uint32_t nbins) {
    qu_ = qu;
    if (nbins_ != nbins) {
      nbins_ = nbins;
      data_.clear();
    }
  }

  // Create an empty histogram for i-th node
  void AddHistRow(bst_uint nid) {
    if (data_.count(nid) == 0) {
      data_[nid] =
        std::make_unique<GHistRowT>(qu_, nbins_);
    }
  }

  void PushPointersToDevice() {
    std::vector<GradientPair*> ptrs_host(data_.size(), nullptr);
    for (const auto& [nid, hist] : data_) {
      if (ptrs_host.size() <= nid) ptrs_host.resize(nid + 1);

      ptrs_host[nid] = hist->Data();
    }

    ptrs_.Init(qu_, ptrs_host);
  }

  GradientPair** GetDevicePointers() {
    return ptrs_.Data();
  }

 private:
  /*! \brief Number of all bins over all features */
  uint32_t nbins_ = 0;

  std::unordered_map<uint32_t, std::unique_ptr<GHistRowT>> data_;
  USMVector<GradientPair*, MemoryType::on_device> ptrs_;

  ::sycl::queue* qu_;
};

/*!
 * \brief Stores temporary histograms to compute them in parallel
 */
template<typename GradientSumT>
class ParallelGHistBuilder {
 public:
  using GHistRowT = GHistRow<GradientSumT, MemoryType::on_device>;

  void Init(::sycl::queue* qu, size_t nbins) {
    qu_ = qu;
    if (nbins != nbins_) {
      nbins_ = nbins;

      size_t cache_line_size = 64;
      size_t hist_size = 2 * nbins_ * sizeof(GradientSumT);
      block_size_ = (hist_size / cache_line_size + (hist_size % cache_line_size > 0)) * cache_line_size / (2 * sizeof(GradientSumT));
    }
  }

  void Reset(size_t nblocks) {
    hist_device_buffer_.Resize(qu_, nblocks * block_size_);
  }

  size_t GetNBlocks() const {
    return hist_device_buffer_.Size() / block_size_;
  }

  GHistRowT& GetDeviceBuffer() {
    return hist_device_buffer_;
  }

 protected:
  /*! \brief Number of bins in each histogram */
  size_t nbins_ = 0;
  size_t block_size_ = 0;

  /*! \brief Buffer for additional histograms for Parallel processing  */
  GHistRowT hist_device_buffer_;

  ::sycl::queue* qu_;
};

/*!
 * \brief Builder for histograms of gradient statistics
 */
template<typename GradientSumT>
class GHistBuilder {
 public:
  template<MemoryType memory_type = MemoryType::shared>
  using GHistRowT = GHistRow<GradientSumT, memory_type>;

  GHistBuilder() = default;
  GHistBuilder(::sycl::queue* qu, uint32_t nbins) : qu_{qu}, nbins_{nbins} {}

  // Construct a histogram via histogram aggregation
  ::sycl::event BuildHist(const HostDeviceVector<GradientPair>& gpair,
                          const std::vector<bst_node_t>& nodes,
                          const bst_node_t* nodes_device_ptr,
                          RowSetCollection* row_indices,
                          const GHistIndexMatrix& gmat,
                          HistCollection<GradientSumT>* histograms,
                          GHistRowT<MemoryType::on_device>* hist_buffer,
                          const DeviceProperties& device_prop,
                          ::sycl::event event,
                          bool force_atomic_use = false);

  // Construct a histogram via histogram aggregation
  ::sycl::event BuildHist(const HostDeviceVector<GradientPair>& gpair,
                          const std::vector<bst_node_t>& nodes,
                          const bst_node_t* nodes_device_ptr,
                          RowSetCollection* row_indices,
                          const GHistIndexMatrix& gmat,
                          HistCollection<GradientSumT>* histograms,
                          const DeviceProperties& device_prop,
                          ::sycl::event event,
                          bool force_atomic_use = false);

  // Construct a histogram via histogram aggregation
  ::sycl::event BuildHistL1(const HostDeviceVector<GradientPair>& gpair,
                          const std::vector<bst_node_t>& nodes,
                          const bst_node_t* nodes_device_ptr,
                          RowSetCollection* row_indices,
                          const GHistIndexMatrix& gmat,
                          HistCollection<GradientSumT>* histograms,
                          const DeviceProperties& device_prop,
                          ::sycl::event event);

  uint32_t GetNumBins() const {
      return nbins_;
  }

 private:
  /*! \brief Number of all bins over all features */
  uint32_t nbins_ { 0 };

  ::sycl::queue* qu_;
};
}  // namespace common
}  // namespace sycl
}  // namespace xgboost
#endif  // PLUGIN_SYCL_COMMON_HIST_UTIL_H_
