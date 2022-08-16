/*!
 * Copyright 2017-2022 by Contributors
 * \file column_matrix.h
 * \brief Utility for fast column-wise access
 * \author Philip Cho
 */

#ifndef XGBOOST_COMMON_COLUMN_MATRIX_H_
#define XGBOOST_COMMON_COLUMN_MATRIX_H_

#include <dmlc/endian.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>
#include <utility>

#include "../data/gradient_index.h"
#include "hist_util.h"

namespace xgboost {
namespace common {

/*! \brief column type */
enum ColumnType : uint8_t { kDenseColumn, kSparseColumn };

/*! \brief a column storage, to be used with ApplySplit. Note that each
    bin id is stored as index[i] + index_base.
    Different types of column index for each column allow
    to reduce the memory usage. */
class Column {
 public:
  using ByteType = uint8_t;
  using BinCmpType = int32_t;
  static constexpr BinCmpType kMissingId = -1;

  Column(ColumnType type, BinTypeSize bin_type_size,
         common::Span<const ByteType> index, const uint32_t index_base)
      : type_(type),
        bin_type_size_(bin_type_size),
        index_(index),
        index_base_(index_base) {}

  virtual ~Column() = default;

  uint32_t GetGlobalBinIdx(size_t idx) const {
    uint32_t res = index_base_;
    if (GetBinTypeSize() == kUint8BinsTypeSize) {
      res += GetFeatureBinIdx<BinTypeMap<kUint8BinsTypeSize>::Type>(idx);
    } else if (GetBinTypeSize() == kUint16BinsTypeSize) {
      res += GetFeatureBinIdx<BinTypeMap<kUint16BinsTypeSize>::Type>(idx);
    } else {
      res += GetFeatureBinIdx<BinTypeMap<kUint32BinsTypeSize>::Type>(idx);
    }
    return res;
  }

  template <typename BinIdxType>
  BinIdxType GetFeatureBinIdx(size_t idx) const {
    const BinIdxType * ptr = reinterpret_cast<const BinIdxType *>(index_.data());
    return ptr[idx];
  }

  uint32_t GetBaseIdx() const { return index_base_; }

  template <typename BinIdxType>
  common::Span<const BinIdxType> GetFeatureBinIdxPtr() const {
    return { reinterpret_cast<BinIdxType>(index_), index_.size() / sizeof(BinIdxType)};
  }

  ColumnType GetType() const { return type_; }

  BinTypeSize GetBinTypeSize() const { return bin_type_size_; }

  /* returns number of elements in column */
  size_t Size() const { return index_.size() / bin_type_size_; }

 private:
  /* type of column */
  ColumnType type_;
  /* size of bin type idx*/
  BinTypeSize bin_type_size_;
  /* bin indexes in range [0, max_bins - 1] */
  common::Span<const ByteType> index_;
  /* bin index offset for specific feature */
  bst_bin_t const index_base_;
};

class SparseColumn: public Column {
 public:
  SparseColumn(BinTypeSize bin_type_size, common::Span<const ByteType> index,
              uint32_t index_base, common::Span<const size_t> row_ind)
      : Column(ColumnType::kSparseColumn, bin_type_size, index, index_base),
        row_ind_(row_ind) {}

  const size_t* GetRowData() const { return row_ind_.data(); }

  template <typename BinIdxType, typename CastType = Column::BinCmpType>
  CastType GetBinIdx(size_t rid, size_t* state) const {
    const size_t column_size = this->Size();
    if (!((*state) < column_size)) {
      return static_cast<CastType>(this->kMissingId);
    }
    while ((*state) < column_size && GetRowIdx(*state) < rid) {
      ++(*state);
    }
    if (((*state) < column_size) && GetRowIdx(*state) == rid) {
      return static_cast<CastType>(this->GetFeatureBinIdx<BinIdxType>(*state));
    } else {
      return static_cast<CastType>(this->kMissingId);
    }
  }

  Column::BinCmpType operator[](size_t rid) const {
    const size_t column_size = this->Size();
    if (!(state_ < column_size)) {
      return static_cast<Column::BinCmpType>(this->kMissingId);
    }
    while (state_ < column_size && GetRowIdx(state_) < rid) {
      ++state_;
    }
    if ((state_ < column_size) && GetRowIdx(state_) == rid) {
      return static_cast<Column::BinCmpType>(this->Column::GetGlobalBinIdx(state_));
    } else {
      return static_cast<Column::BinCmpType>(this->kMissingId);
    }
  }

  Column::BinCmpType GetGlobalBinIdx(size_t idx) const {
    return this->Column::GetGlobalBinIdx(idx);
  } 

  size_t GetInitialState(const size_t first_row_id) const {
    const size_t* row_data = GetRowData();
    const size_t column_size = this->Size();
    // search first nonzero row with index >= rid_span.front()
    const size_t* p = std::lower_bound(row_data, row_data + column_size, first_row_id);
    // column_size if all messing
    return p - row_data;
  }

  size_t GetRowIdx(size_t idx) const {
    return row_ind_.data()[idx];
  }

 private:
  mutable size_t state_ = 0;
  /* indexes of rows */
  common::Span<const size_t> row_ind_;
};

class DenseColumn: public Column {
 public:
  DenseColumn(BinTypeSize bin_type_size, common::Span<const ByteType> index,
              uint32_t index_base, const bool any_missing,
              const std::vector<ByteType>& missing_flags,
              size_t feature_offset)
      : Column(ColumnType::kDenseColumn, bin_type_size, index, index_base),
        any_missing_(any_missing),
        missing_flags_(missing_flags),
        feature_offset_(feature_offset) {}
  bool IsMissing(size_t idx) const {
  const bool res = missing_flags_[feature_offset_ + idx];
    return res;}

  template <typename BinIdxType, typename CastType = Column::BinCmpType>
  CastType GetBinIdx(size_t idx, size_t* state) const {
    return static_cast<CastType>(
              (any_missing_ && IsMissing(idx))
              ? this->kMissingId
              : this->GetFeatureBinIdx<BinIdxType>(idx));
  }

  Column::BinCmpType operator[](size_t rid) const {
      return static_cast<Column::BinCmpType>(
              (any_missing_ && IsMissing(rid))
              ? this->kMissingId
              : this->GetGlobalBinIdx(rid));
  }

  Column::BinCmpType GetGlobalBinIdx(size_t idx) const {
    return this->Column::GetGlobalBinIdx(idx);
  }

  size_t GetInitialState(const size_t first_row_id) const { return 0; }

 private:
  const bool any_missing_;
  /* flags for missing values in dense columns */
  const std::vector<ByteType>& missing_flags_;
  size_t feature_offset_;
};

class ColumnView final {
 public:
  ColumnView() = delete;
  explicit ColumnView(const SparseColumn * sparse_clmn_ptr) :
    sparse_clmn_ptr_(sparse_clmn_ptr),
    dense_clmn_ptr_(nullptr) { }
  explicit ColumnView(const DenseColumn * dense_clmn_ptr) :
    sparse_clmn_ptr_(nullptr),
    dense_clmn_ptr_(dense_clmn_ptr) { }

  uint32_t GetGlobalBinIdx(size_t idx) const {
    return sparse_clmn_ptr_ ? sparse_clmn_ptr_->GetGlobalBinIdx(idx)
                            : dense_clmn_ptr_->GetGlobalBinIdx(idx);
  }

  uint32_t operator[] (size_t rid) const {
    return sparse_clmn_ptr_ ? (*sparse_clmn_ptr_)[rid]
                            : (*dense_clmn_ptr_)[rid];
  }

  template <typename BinIdxType>
  BinIdxType GetFeatureBinIdx(size_t idx) const {
    return sparse_clmn_ptr_ ? sparse_clmn_ptr_->GetFeatureBinIdx<BinIdxType>(idx)
                            : dense_clmn_ptr_->GetFeatureBinIdx<BinIdxType>(idx);
  }

  uint32_t GetBaseIdx() const {
    return sparse_clmn_ptr_ ? sparse_clmn_ptr_->GetBaseIdx()
                            : dense_clmn_ptr_->GetBaseIdx();
  }

  template <typename BinIdxType>
  common::Span<const BinIdxType> GetFeatureBinIdxPtr() const {
    return sparse_clmn_ptr_ ? sparse_clmn_ptr_->GetFeatureBinIdxPtr<BinIdxType>()
                            : dense_clmn_ptr_->GetFeatureBinIdxPtr<BinIdxType>();
  }

  ColumnType GetType() const {
    return sparse_clmn_ptr_ ? ColumnType::kSparseColumn : ColumnType::kDenseColumn;
  }

  size_t Size() const {
    return sparse_clmn_ptr_ ? sparse_clmn_ptr_->Size() : dense_clmn_ptr_->Size();
  }

  const size_t* GetRowData() const {
    return sparse_clmn_ptr_ ? sparse_clmn_ptr_->GetRowData() : nullptr;
  }

  template <typename BinIdxType, typename CastType = Column::BinCmpType>
  CastType GetBinIdx(size_t rid, size_t* state) const {
    return sparse_clmn_ptr_ ? sparse_clmn_ptr_->GetBinIdx<BinIdxType, CastType>(rid, state)
                            : dense_clmn_ptr_->GetBinIdx<BinIdxType, CastType>(rid, state);
  }

  size_t GetInitialState(const size_t first_row_id) const {
    return sparse_clmn_ptr_ ? sparse_clmn_ptr_->GetInitialState(first_row_id)
                            : dense_clmn_ptr_->GetInitialState(first_row_id);
  }

  size_t GetRowIdx(size_t idx) const {
    return sparse_clmn_ptr_ ? sparse_clmn_ptr_->GetRowIdx(idx) : 0;
  }

 private:
  const SparseColumn * sparse_clmn_ptr_;
  const DenseColumn* dense_clmn_ptr_;
};

/*! \brief a collection of columns, with support for construction from
    GHistIndexMatrix. */
class ColumnMatrix {
  void InitStorage(GHistIndexMatrix const& gmat, double sparse_threshold);

 public:
  using ByteType = uint8_t;
  using ColumnListType = std::vector<std::shared_ptr<const Column>>;
  using ColumnViewListType = std::vector<std::shared_ptr<const ColumnView>>;
  // get number of features
  bst_feature_t GetNumFeature() const { return static_cast<bst_feature_t>(type_.size()); }
  // get index data ptr
  template <typename Data>
  const Data* GetIndexData() const {
    return reinterpret_cast<const Data*>(index_.data());
  }

  // get index data ptr
  const ByteType* GetIndexData() const {
    return index_.data();
  }

  ColumnMatrix() = default;
  ColumnMatrix(GHistIndexMatrix const& gmat, double sparse_threshold) {
    this->InitStorage(gmat, sparse_threshold);
  }

  template <typename Batch>
  void PushBatch(int32_t n_threads, Batch const& batch, GHistIndexMatrix const& gmat,
                 size_t base_rowid) {
    // pre-fill index_ for dense columns
    auto n_features = gmat.Features();
    if (all_dense_column_) {
      missing_flags_.resize(feature_offsets_[n_features], false);
      // row index is compressed, we need to dispatch it.
      DispatchBinType(gmat.index.GetBinTypeSize(), [&](auto t) {
        using RowBinIdxT = decltype(t);
        SetIndexAllDense<RowBinIdxT>(base_rowid, gmat, batch.Size(), n_threads);
      });
    /* For sparse DMatrix gmat.index.getBinTypeSize() returns always kUint32BinsTypeSize
     * but for ColumnMatrix we still have a chance to reduce the memory consumption
     */
    } else {
      missing_flags_.resize(feature_offsets_[n_features], true);
      DispatchBinType(bin_type_size_, [&](auto t) {
        using ColumnBinT = decltype(t);
        SetIndex<ColumnBinT>(batch, gmat);
      });
    }
    FillColumnViewList(n_features);
  }

  const ColumnListType& GetColumnList() const { return column_list_; }
  const ColumnViewListType& GetColumnViewList() const { return column_view_list_; }

  // construct column matrix from GHistIndexMatrix
  void Init(SparsePage const& page, const GHistIndexMatrix& gmat, double sparse_threshold,
            int32_t n_threads) {
    auto batch = data::SparsePageAdapterBatch{page.GetView()};
    this->InitStorage(gmat, sparse_threshold);
    // ignore base row id here as we always has one column matrix for each sparse page.
    this->PushBatch(n_threads, batch, gmat, 0);
  }

  /* Set the number of bytes based on numeric limit of maximum number of bins provided by user */
  void SetTypeSize(size_t max_num_bins) {
    if ((max_num_bins - 1) <= static_cast<int>(std::numeric_limits<uint8_t>::max())) {
      bin_type_size_ = kUint8BinsTypeSize;
    } else if ((max_num_bins - 1) <= static_cast<int>(std::numeric_limits<uint16_t>::max())) {
      bin_type_size_ = kUint16BinsTypeSize;
    } else {
      bin_type_size_ = kUint32BinsTypeSize;
    }
  }

  template <typename RowBinIdxT>
  inline void SetIndexAllDense(bst_row_t base_rowid,
                               const GHistIndexMatrix& gmat,
                               size_t n_row,
                               int32_t n_threads) {
    const RowBinIdxT* index = gmat.index.data<RowBinIdxT>();
    const size_t n_features = gmat.Features();
    RowBinIdxT* local_index = reinterpret_cast<RowBinIdxT*>(&index_[0]);

    /* missing values make sense only for column with type kDenseColumn,
       and if no missing values were observed it could be handled much faster. */
    ParallelFor(n_row, n_threads, [&](auto rid) {
      rid += base_rowid;
      const size_t ibegin = rid * n_features;
      const size_t iend = (rid + 1) * n_features;
      size_t j = 0;
      for (size_t i = ibegin; i < iend; ++i, ++j) {
        const size_t idx = feature_offsets_[j];
        local_index[idx + rid] = index[i];
      }
    });
  }

  template <typename T, typename BinFn, typename Batch>
  void SetIndexSparse(Batch const& batch, T* index, const GHistIndexMatrix& gmat,
                      const size_t n_feature, BinFn&& assign_bin) {
    auto rbegin = 0;
    size_t const batch_size = batch.Size();

    for (size_t rid = 0; rid < batch_size; ++rid) {
      auto line = batch.GetLine(rid);
      const size_t ibegin = gmat.row_ptr[rbegin + rid];
      const size_t iend = gmat.row_ptr[rbegin + rid + 1];
      for (size_t i = 0; i < line.Size(); ++i) {
        if (i + ibegin < iend) {
          auto coo = line.GetElement(i);
          auto fid = coo.column_idx;
          const uint32_t bin_id = index[i + ibegin];
          assign_bin(bin_id, rid, fid);
        }
      }
    }
  }

  template <typename ColumnBinT, typename Batch>
  inline void SetIndex(Batch const& batch, const GHistIndexMatrix& gmat) {
    const uint32_t* index = gmat.index.data<uint32_t>();
    ColumnBinT* local_index = reinterpret_cast<ColumnBinT*>(&index_[0]);

    const size_t n_features = gmat.Features();
    std::vector<size_t> num_nonzeros;
    num_nonzeros.resize(n_features);
    std::fill(num_nonzeros.begin(), num_nonzeros.end(), 0);

    auto get_bin_idx = [&](auto bin_id, auto rid, bst_feature_t fid) {
      if (type_[fid] == kDenseColumn) {
        ColumnBinT* begin = &local_index[feature_offsets_[fid]];
        begin[rid] = bin_id - index_base_[fid];
        missing_flags_[feature_offsets_[fid] + rid] = false;
      } else {
        ColumnBinT* begin = &local_index[feature_offsets_[fid]];
        begin[num_nonzeros[fid]] = bin_id - index_base_[fid];
        row_ind_[feature_offsets_[fid] + num_nonzeros[fid]] = rid;
        ++num_nonzeros[fid];
      }
    };
    this->SetIndexSparse(batch, index, gmat, n_features, get_bin_idx);
  }

  BinTypeSize GetTypeSize() const {
    return bin_type_size_;
  }

  size_t GetSizeMissing() const {
    return missing_flags_.size();
  }

  const std::vector<ByteType>* GetMissing() const {
    return &missing_flags_;
  }

  // This is just an utility function
  bool NoMissingValues(const size_t n_element, const size_t n_row, const size_t n_feature) {
    return n_element == n_feature * n_row;
  }

  // And this returns part of state
  bool AnyMissing() const { return any_missing_; }

  // IO procedures for external memory.
  bool Read(dmlc::SeekStream* fi, uint32_t const* index_base) {
    fi->Read(&index_);
#if !DMLC_LITTLE_ENDIAN
    // s390x
    std::vector<std::underlying_type<ColumnType>::type> int_types;
    fi->Read(&int_types);
    type_.resize(int_types.size());
    std::transform(
        int_types.begin(), int_types.end(), type_.begin(),
        [](std::underlying_type<ColumnType>::type i) { return static_cast<ColumnType>(i); });
#else
    fi->Read(&type_);
#endif  // !DMLC_LITTLE_ENDIAN

    fi->Read(&row_ind_);
    fi->Read(&feature_offsets_);
    fi->Read(&missing_flags_);
    index_base_ = index_base;
#if !DMLC_LITTLE_ENDIAN
    std::underlying_type<BinTypeSize>::type v;
    fi->Read(&v);
    bin_type_size_ = static_cast<BinTypeSize>(v);
#else
    fi->Read(&bin_type_size_);
#endif
    fi->Read(&any_missing_);

    FillColumnViewList(type_.size());

    return true;
  }

  size_t Write(dmlc::Stream* fo) const {
    size_t bytes{0};

    auto write_vec = [&](auto const& vec) {
      fo->Write(vec);
      bytes += vec.size() * sizeof(typename std::remove_reference_t<decltype(vec)>::value_type) +
               sizeof(uint64_t);
    };
    write_vec(index_);
#if !DMLC_LITTLE_ENDIAN
    // s390x
    std::vector<std::underlying_type<ColumnType>::type> int_types(type_.size());
    std::transform(type_.begin(), type_.end(), int_types.begin(), [](ColumnType t) {
      return static_cast<std::underlying_type<ColumnType>::type>(t);
    });
    write_vec(int_types);
#else
    write_vec(type_);
#endif  // !DMLC_LITTLE_ENDIAN
    write_vec(row_ind_);
    write_vec(feature_offsets_);
    write_vec(missing_flags_);
#if !DMLC_LITTLE_ENDIAN
    auto v = static_cast<std::underlying_type<BinTypeSize>::type>(bin_type_size_);
    fo->Write(v);
#else
    fo->Write(bin_type_size_);
#endif  // DMLC_LITTLE_ENDIAN
    bytes += sizeof(bin_type_size_);
    fo->Write(any_missing_);
    bytes += sizeof(any_missing_);

    return bytes;
  }

  const size_t* GetRowId() const {
    return row_ind_.data();
  }

 private:
  template <typename ColumnType, typename ... Args>
  void AddColumnToList(size_t fid, Args&& ... args) {
        auto clmn = std::make_shared<const ColumnType>(std::forward<Args>(args) ...);
        column_list_[fid] = clmn;
        column_view_list_[fid] = std::make_shared<const ColumnView>(clmn.get());
  }

  /* Filling list of helpers for operating with columns */
  void FillColumnViewList(const size_t n_feature) {
    column_list_.resize(n_feature);
    column_view_list_.resize(n_feature);
    for (auto fid = 0; fid < n_feature; ++fid) {
      // to get right place for certain feature
      const size_t feature_offset = feature_offsets_[fid];
      const size_t column_size = feature_offsets_[fid + 1] - feature_offset;
      common::Span<const ByteType> bin_index = { &index_[feature_offset * bin_type_size_],
                                                   column_size * bin_type_size_ };

      if (type_[fid] == ColumnType::kDenseColumn) {
        AddColumnToList<DenseColumn>(fid, GetTypeSize(), bin_index,
                              index_base_[fid],
                              any_missing_, missing_flags_, feature_offset);
      } else {
        AddColumnToList<SparseColumn>(fid, GetTypeSize(), bin_index,
                              index_base_[fid],
                              common::Span<const size_t>(&row_ind_[feature_offset], column_size));
      }
    }
  }

 private:
  std::vector<ByteType> index_;

  std::vector<ColumnType> type_;
  /* indptr of a CSC matrix. */
  std::vector<size_t> row_ind_;
  /* indicate where each column's index and row_ind is stored. */
  std::vector<size_t> feature_offsets_;
  /* The number of nnz of each column. */
  std::vector<size_t> num_nonzeros_;

  // index_base_[fid]: least bin id for feature fid
  uint32_t const* index_base_ = nullptr;
  std::vector<ByteType> missing_flags_;
  BinTypeSize bin_type_size_ = static_cast<BinTypeSize>(0);
  bool any_missing_;
  bool all_dense_column_;
  common::HistogramCuts cut_;

  ColumnListType column_list_;
  ColumnViewListType column_view_list_;
};
}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_COLUMN_MATRIX_H_
