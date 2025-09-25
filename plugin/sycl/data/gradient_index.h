/*!
 * Copyright 2017-2024 by Contributors
 * \file gradient_index.h
 */
#ifndef PLUGIN_SYCL_DATA_GRADIENT_INDEX_H_
#define PLUGIN_SYCL_DATA_GRADIENT_INDEX_H_

#include <vector>

#include "../data.h"
#include "../../src/common/hist_util.h"
#include "../../src/common/ref_resource_view.h"

#include <sycl/sycl.hpp>

namespace xgboost {
namespace sycl {
namespace common {

using BinTypeSize = ::xgboost::common::BinTypeSize;

/*!
 * \brief Index data and offsets stored in USM buffers to provide access from device kernels
 */
struct Index {
  Index() {
    SetBinTypeSize(binTypeSize_);
  }
  Index(const Index& i) = delete;
  Index& operator=(Index i) = delete;
  Index(Index&& i) = delete;
  Index& operator=(Index&& i) = delete;
  void SetBinTypeSize(BinTypeSize binTypeSize) {
    binTypeSize_ = binTypeSize;
    CHECK(binTypeSize == BinTypeSize::kUint8BinsTypeSize  ||
          binTypeSize == BinTypeSize::kUint16BinsTypeSize ||
          binTypeSize == BinTypeSize::kUint32BinsTypeSize);
  }
  BinTypeSize GetBinTypeSize() const {
    return binTypeSize_;
  }

  template<typename T>
  T* data() {
    return reinterpret_cast<T*>(data_.Data());
  }

  template<typename T>
  const T* data() const {
    return reinterpret_cast<const T*>(data_.DataConst());
  }

  size_t Size() const {
    return data_.Size() / (binTypeSize_);
  }

  void Resize(::sycl::queue* qu, const size_t nBytesData) {
    data_.ResizeNoCopy(qu, nBytesData);
  }

  uint8_t* begin() {
    return data_.Begin();
  }

  uint8_t* end() {
    return data_.End();
  }

  USMVector<uint8_t, MemoryType::on_device> data_;
 private:
  BinTypeSize binTypeSize_ {BinTypeSize::kUint8BinsTypeSize};
};

/*!
 * \brief Preprocessed global index matrix, in CSR format, stored in USM buffers
 *
 *  Transform floating values to integer index in histogram
 */
struct GHistIndexMatrix {
  /*! \brief row pointer to rows by element position */
  /*! \brief The index data */
  Index index;
  /*! \brief hit count of each index */
  xgboost::common::Span<const std::size_t> hit_count;

  // USMVector<uint8_t, MemoryType::on_device> sort_buff;
  /*! \brief The corresponding cuts */
  xgboost::common::HistogramCuts cut;
  std::shared_ptr<xgboost::common::HistogramCuts> cut_device;
  size_t max_num_bins;
  size_t min_num_bins;
  size_t nbins;
  size_t nfeatures;
  size_t row_stride;
  size_t n_rows;
  size_t base_rowid = 0;
  int page_idx = -1;

  // Create a global histogram matrix based on a given DMatrix device wrapper
  void Init(::sycl::queue* qu, Context const * ctx,
            DMatrix *dmat, int max_num_bins);

  void Init(::sycl::queue* qu,
            Context const * ctx,
            const xgboost::GHistIndexMatrix& page,
            std::shared_ptr<xgboost::common::HistogramCuts> cut,
            size_t max_num_bins,
            size_t min_num_bins,
            int page_idx);

  template <typename BinIdxType, bool isDense>
  void SetIndexData(::sycl::queue* qu, Context const * ctx, BinIdxType* index_data,
                    DMatrix *dmat);

  void ResizeIndex(::sycl::queue* qu, size_t n_index);

  inline void GetFeatureCounts(size_t* counts) const {
    auto nfeature = cut.cut_ptrs_.Size() - 1;
    for (unsigned fid = 0; fid < nfeature; ++fid) {
      auto ibegin = cut.cut_ptrs_.ConstHostVector()[fid];
      auto iend = cut.cut_ptrs_.ConstHostVector()[fid + 1];
      for (auto i = ibegin; i < iend; ++i) {
        *(counts + fid) += hit_count[i];
      }
    }
  }
  inline bool IsDense() const {
    return isDense_;
  }

 private:
  bool isDense_;
};

class GHistIndexCache {
  size_t n_pages_;
  std::vector<std::unique_ptr<GHistIndexMatrix>> gmat_cache_;
  std::vector<int> cached_pages_;
  std::vector<int> computed_pages_;
  size_t global_mem_size_;
  ::sycl::queue* qu_;
  size_t index_size_;
  std::shared_ptr<xgboost::common::HistogramCuts> cut_;

 public:
  void Init(::sycl::queue* qu, std::shared_ptr<xgboost::common::HistogramCuts> cut,
            size_t n_pages, size_t index_size) {
    n_pages_ = n_pages;
    qu_ = qu;
    cut_ = cut;
    gmat_cache_.resize(n_pages);
    computed_pages_.resize(n_pages);
    index_size_ = index_size;
    global_mem_size_ = qu_->get_device().get_info<::sycl::info::device::global_mem_size>();

    size_t max_mem_alloc_size = qu_->get_device().get_info<::sycl::info::device::max_mem_alloc_size>();
    CHECK_LT(index_size_, max_mem_alloc_size) << "Can't allocate gradient index as a single memory block";
  }

  size_t NumCachedPages() const {
    size_t num_cached_pages = 0;
    for (size_t page_idx = 0; page_idx < n_pages_; ++page_idx) {
      num_cached_pages += (gmat_cache_[page_idx].get() != nullptr);
    }
    return num_cached_pages;
  }

  size_t MaxCacheSize(size_t reserved_mem) const {
    size_t max_cache_size = 0;
    // factor 0.9 for some minor memory allocations not taken into accout in reserved_mem
    // free_memory is reduced by index_size_ to prevent fragmentation issues
    double free_memory = (0.9 * global_mem_size_) - reserved_mem;
    if (free_memory > 0) {
      max_cache_size = free_memory / index_size_;
    }

    return max_cache_size;
  }

  void ShrinkCache(size_t max_cache_size) {
    while (cached_pages_.size() > max_cache_size) {
      // fprintf(stderr, "Removing page %d from cache\n", cached_pages_.back());
      gmat_cache_[cached_pages_.back()].reset();
      cached_pages_.pop_back();
    }
  }

  void AllocatePage(size_t page_idx, size_t max_cache_size) {
    // Allocate new page
    if (cached_pages_.size() < max_cache_size) {
      gmat_cache_[page_idx] = std::make_unique<GHistIndexMatrix>();
      cached_pages_.push_back(page_idx);
      // fprintf(stderr, "Adding page %d to cache\n", cached_pages_.back());
      return;
    }

    // Deallocate pages
    if (cached_pages_.size() > max_cache_size) {
      ShrinkCache(max_cache_size);
    }

    // cached_pages_.size() == max_cache_size
    // move memory to another page
    // fprintf(stderr, "Mooving page %d to %ld\n", cached_pages_.back(), page_idx);
    gmat_cache_[page_idx] = std::move(gmat_cache_[cached_pages_.back()]);
    CHECK_EQ(gmat_cache_[cached_pages_.back()].get(), nullptr);
    cached_pages_.back() = page_idx;
  }

  template <class Fn>
  void Process(const Context* ctx, const BatchParam& batch_params, DMatrix *p_fmat, 
               size_t max_num_bins, size_t min_num_bins, size_t max_cache_size, Fn&& fn) {
    std::fill(computed_pages_.begin(), computed_pages_.end(), 0);
    for (size_t page_idx = 0; page_idx < n_pages_; ++page_idx) {
      if (gmat_cache_[page_idx].get() != nullptr) {
        fn(*gmat_cache_[page_idx]);
        computed_pages_[page_idx] = 1;
      }
    }

    size_t page_idx = 0;

    CHECK_GE(max_cache_size, 1) << "Out of memory on device";

    for (auto const &page : p_fmat->GetBatches<xgboost::GHistIndexMatrix>(ctx, batch_params)) {
      if (computed_pages_[page_idx] == 0) {
        AllocatePage(page_idx, max_cache_size);
        gmat_cache_[page_idx]->Init(qu_, ctx, page, cut_, max_num_bins, min_num_bins,
                                    page_idx);
        fn(*gmat_cache_[page_idx]);
      }
      page_idx++;
    }
  }
};


}  // namespace common
}  // namespace sycl
}  // namespace xgboost
#endif  // PLUGIN_SYCL_DATA_GRADIENT_INDEX_H_
