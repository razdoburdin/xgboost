
/*!
 * Copyright 2022 by Contributors
 * \file opt_partition_builder.h
 * \brief Quick Utility to compute subset of rows
 */
#ifndef XGBOOST_COMMON_OPT_PARTITION_BUILDER_H_
#define XGBOOST_COMMON_OPT_PARTITION_BUILDER_H_

#include <xgboost/data.h>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <utility>
#include <memory>
#include "xgboost/tree_model.h"
#include "column_matrix.h"
#include "threads_manager.h"
#include "flexible_container.h"
#include "../tree/hist/expand_entry.h"

namespace xgboost {
namespace common {

struct RowIndicesRange {
  size_t begin;
  size_t end;
};

struct SplitNode {
  int32_t condition = 0;
  uint64_t ind = 0;
  bool smalest_nodes_mask = false;
};

// The builder is required for samples partition to left and rights children for set of nodes
// template by number of rows
class OptPartitionBuilder {
  size_t depth_;
  uint16_t* node_ids_;

 public:
  ThreadsManager tm;
  std::vector<Slice> partitions;
  const RegTree* p_tree;
  // can be common for all threads!
  const uint8_t* data_hash;
  std::vector<bool>* missing_ptr;
  size_t* row_ind_ptr;
  std::vector<size_t> row_set_collection_vec;
  uint32_t gmat_n_rows;
  uint32_t base_rowid;
  size_t* row_indices_ptr;
  size_t n_threads = 0;
  uint32_t summ_size = 0;
  uint32_t summ_size_remain = 0;
  uint32_t max_depth = 0;

  static constexpr double adhoc_l2_size = 1024 * 1024 * 0.8;
  static constexpr uint32_t thread_size_limit = 512;

  static bool use_linear_containers(size_t max_depth) {
  /* We use vector insteard of unordered_map to improve performance.
   * Unfortunately vector can't be used in case of larger depth due to memory limitations.
   * Maximal depth = 16 is an adhock parameter.
   */
    return (max_depth > 0) && (max_depth <= 16);
  }

  const std::vector<uint16_t> &GetThreadIdsForNode(const uint32_t nid) const {
    auto node_info = tm.GetNodeInfoPtr(nid);
    const std::vector<uint16_t> & res = node_info->threads_id;
    return res;
  }

  void Init(const ColumnMatrix& column_matrix,
            GHistIndexMatrix const& gmat,
            const RegTree* p_tree_local, size_t nthreads, size_t max_depth,
            bool is_loss_guided) {
    gmat_n_rows = gmat.row_ptr.size() - 1;
    base_rowid = gmat.base_rowid;
    p_tree = p_tree_local;
    if ((tm.NumThreads() == 0 && column_matrix.AnyMissing()) ||
        (data_hash != reinterpret_cast<const uint8_t*>(column_matrix.GetIndexData())
        && column_matrix.AnyMissing()) ||
        (missing_ptr != column_matrix.GetMissing() && column_matrix.AnyMissing()) ||
        (row_ind_ptr != column_matrix.GetRowId())) {
      missing_ptr = const_cast<std::vector<bool>*>(column_matrix.GetMissing());
      row_ind_ptr = const_cast<size_t*>(column_matrix.GetRowId());
    }
    data_hash = reinterpret_cast<const uint8_t*>(column_matrix.GetIndexData());
    this->max_depth = max_depth;
    if (is_loss_guided) {
      partitions.resize(nodes_amount(max_depth));
    }

    n_threads = nthreads;
    size_t chunck_size = common::GetBlockSize(gmat_n_rows, nthreads);
    tm.Init(n_threads, chunck_size, is_loss_guided,
            use_linear_containers(max_depth), nodes_amount(max_depth));
    UpdateRootThreadWork();
  }

  template<bool is_loss_guided, bool all_dense, bool any_cat, typename Predicate>
  void CommonPartition(const ColumnMatrix& column_matrix, Predicate&& pred, size_t tid,
                       const RowIndicesRange& row_indices,
                       const FlexibleContainer<SplitNode>& split_info) {
    switch (column_matrix.GetTypeSize()) {
      case common::kUint8BinsTypeSize:
        CommonPartition<BinTypeMap<kUint8BinsTypeSize>::Type, is_loss_guided, all_dense, any_cat>(
                           column_matrix, std::forward<Predicate>(pred),
                           column_matrix.template GetIndexData<uint8_t>(),
                           tid, row_indices, split_info);
        break;
      case common::kUint16BinsTypeSize:
        CommonPartition<BinTypeMap<kUint16BinsTypeSize>::Type, is_loss_guided, all_dense, any_cat>(
                           column_matrix, std::forward<Predicate>(pred),
                           column_matrix.template GetIndexData<uint16_t>(),
                           tid, row_indices, split_info);
        break;
      default:
        CommonPartition<BinTypeMap<kUint32BinsTypeSize>::Type, is_loss_guided, all_dense, any_cat>(
                           column_matrix, std::forward<Predicate>(pred),
                           column_matrix.template GetIndexData<uint32_t>(),
                           tid, row_indices, split_info);
    }
  }

  void SetDepth(size_t depth) {
    depth_ = depth;
  }

  void SetNodeIdsPtr(uint16_t* node_ids) {
    node_ids_ = node_ids;
  }

  static size_t nodes_amount(size_t depth) {
    // 2^(depth + 1)
    constexpr size_t one = 1;
    return one << (depth + 1);
  }


  template<typename BinIdxType, bool is_loss_guided,
           bool all_dense, bool any_cat, typename Predicate>
    void CommonPartition(const ColumnMatrix& column_matrix, Predicate&& pred,
                          const BinIdxType* numa, size_t tid,
                          const RowIndicesRange& row_indices,
                          const FlexibleContainer<SplitNode>& split_info) {
      if (split_info.GetContainerType() == ContainerType::kVector) {
        CommonPartition<BinIdxType, is_loss_guided, all_dense, any_cat, vector_t>(
                           column_matrix, std::forward<Predicate>(pred),
                           numa, tid, row_indices, split_info);
      } else {
        CommonPartition<BinIdxType, is_loss_guided, all_dense, any_cat, unordered_map_t>(
                    column_matrix, std::forward<Predicate>(pred),
                    numa, tid, row_indices, split_info);
      }
    }

  template<typename BinIdxType, bool is_loss_guided,
           bool all_dense, bool any_cat, typename Container,
           typename Predicate>
    void CommonPartition(const ColumnMatrix& column_matrix, Predicate&& pred,
                         const BinIdxType* numa, size_t tid,
                         const RowIndicesRange& row_indices,
                         const FlexibleContainer<SplitNode>& split_info) {
    CHECK_EQ(data_hash, reinterpret_cast<const uint8_t*>(column_matrix.GetIndexData()));
    // const auto& column_list = column_matrix.GetColumnViewList();
    uint32_t rows_count = 0;
    uint32_t rows_left_count = 0;
    auto thread_info = tm.GetThreadInfoPtr(tid);

    size_t* rows = thread_info->vec_rows.data();
    size_t* rows_left = nullptr;
    if (is_loss_guided) {
      rows_left = thread_info->vec_rows_remain.data();
    } else {
      thread_info->nodes_count.ResizeIfSmaller(nodes_amount(depth_));
      thread_info->nodes_count_range.ResizeIfSmaller(nodes_amount(depth_));
    }
    thread_info->states.ResizeIfSmaller(nodes_amount(depth_));
    thread_info->states.Fill(0);

    const BinIdxType* columnar_data = numa;

    const uint32_t first_row_id = !is_loss_guided ? row_indices.begin :
                                                    row_indices_ptr[row_indices.begin];
    for (size_t ii = row_indices.begin; ii < row_indices.end; ++ii) {
      const uint32_t i = !is_loss_guided ? ii : row_indices_ptr[ii];
      const uint32_t nid = node_ids_[i];
      const auto& node = (*p_tree)[nid];
      if (node.IsLeaf()) {
        continue;
      }

      const SplitNode& split_node = split_info.get_element_unsafe(Container(), nid);
      const int32_t sc = split_node.condition;
      const uint64_t si = split_node.ind;

      if (any_cat) {
        const int32_t cmp_value = static_cast<int32_t>(columnar_data[si + i]);
        node_ids_[i] = pred(i, cmp_value, nid, sc) ? node.LeftChild()
                                                   : node.RightChild();
      } else if (all_dense) {
        const int32_t cmp_value = static_cast<int32_t>(columnar_data[si + i]);
        node_ids_[i] = cmp_value <= sc ? node.LeftChild()
                                       : node.RightChild();
      } else {
        size_t* state = &(thread_info->states.get_element_unsafe(Container(), nid));

        int32_t cmp_value = 0;
        if (column_matrix.GetColumnType(si) == xgboost::common::kDenseColumn) {
          auto column = column_matrix.DenseColumn<BinIdxType, !any_cat>(si);
          if (*state == 0) {
            *state = column.GetInitialState(first_row_id);
          }
          cmp_value = column.template GetBinIdx<BinIdxType>(i, state);
        } else {
          auto column = column_matrix.SparseColumn<BinIdxType>(si, row_indices.begin - base_rowid);
          if (*state == 0) {
            *state = column.GetInitialState(first_row_id);
          }
          cmp_value = column.template GetBinIdx<BinIdxType>(i, state);
        }

        if (cmp_value == Column<BinIdxType>::kMissingId) {
          node_ids_[i] =   node.DefaultLeft()
                         ? node.LeftChild()
                         : node.RightChild();
        } else {
          node_ids_[i] = cmp_value <= sc ? node.LeftChild()
                                         : node.RightChild();
        }
      }
      const uint16_t check_node_id = node_ids_[i];
      uint32_t inc = split_info.get_element_unsafe(Container(), check_node_id).smalest_nodes_mask;
      rows[1 + rows_count] = i;
      rows_count += inc;
      if (is_loss_guided) {
        rows_left[1 + rows_left_count] = i;
        rows_left_count += !static_cast<bool>(inc);
      } else {
        thread_info->nodes_count.get_element_unsafe(Container(), check_node_id) += inc;
      }
    }

    rows[0] = rows_count;
    if (is_loss_guided) {
      rows_left[0] = rows_left_count;
    }
  }

  template<bool is_loss_guided>
  size_t DepthSize(GHistIndexMatrix const& gmat,
                   const std::vector<uint16_t>& compleate_trees_depth_wise);

  template <bool is_loss_guided>
  size_t DepthBegin(const std::vector<uint16_t>& compleate_trees_depth_wise);

  void ResizeRowsBuffer(size_t nrows) {
    row_set_collection_vec.resize(nrows);
    row_indices_ptr = row_set_collection_vec.data();
  }

  size_t* GetRowsBuffer() const {
    return row_indices_ptr;
  }

  size_t GetPartitionSize(size_t nid) const {
    return partitions[nid].Size();
  }
  void SetSlice(size_t nid, uint32_t begin, uint32_t size) {
    if (partitions.size()) {
      CHECK_LT(nid, partitions.size());

      partitions[nid].b = begin;
      partitions[nid].e = begin + size;
    }
  }

  void PrepareToUpdateRowBuffer() {
    summ_size = tm.AccumulateVecRows(n_threads);
    summ_size_remain = 0;

    tm.ForEachThread([](auto& ti) {ti.nodes_id.clear();});
    tm.ForEachNode([](auto& ni) {ni.threads_id.clear();});
  }

  bool NeedsBufferUpdate(GHistIndexMatrix const& gmat, size_t n_features) {
    const bool hist_fit_to_l2 = adhoc_l2_size > 2*sizeof(double)*gmat.cut.Ptrs().back();
    const size_t n_bins = gmat.cut.Ptrs().back();

    bool ans = n_features*summ_size / n_threads <
                nodes_amount(depth_)*n_bins ||
                (depth_ >= 1 && !hist_fit_to_l2);
    return ans;
  }

  template <bool is_loss_guided>
  void UpdateRowBuffer(const std::vector<uint16_t>& compleate_trees_depth_wise,
                       GHistIndexMatrix const& gmat, size_t n_features);

  template <bool is_loss_guide>
  void UpdateThreadsWork(const std::vector<uint16_t>& compleate_trees_depth_wise,
                         GHistIndexMatrix const& gmat,
                         size_t n_features,
                         bool is_left_small = true,
                         bool check_is_left_small = false);

  template <bool is_loss_guide, bool needs_buffer_update>
  void UpdateThreadsWork(const std::vector<uint16_t>& compleate_trees_depth_wise);

  void UpdateRootThreadWork() {
    tm.ForEachThread([](auto& ti) {ti.addr.clear();
                                   ti.nodes_id.clear();});
    const uint32_t n_rows = gmat_n_rows;
    const uint32_t block_size = common::GetBlockSize(n_rows, n_threads);
    for (uint32_t tid = 0; tid < n_threads; ++tid) {
      const uint32_t begin = tid * block_size;
      const uint32_t end = std::min(begin + block_size, n_rows);
      if (end > begin) {
        auto thread_info = tm.GetThreadInfoPtr(tid);
        thread_info->addr.push_back({nullptr, begin, end});
        tm.Tie(tid, 0);
      }
    }
  }
};

}  // namespace common
}  // namespace xgboost

#endif  // XGBOOST_COMMON_OPT_PARTITION_BUILDER_H_
