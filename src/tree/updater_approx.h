/*!
 * Copyright 2021 XGBoost contributors
 *
 * \brief Implementation for the approx tree method.
 */
#ifndef XGBOOST_TREE_UPDATER_APPROX_H_
#define XGBOOST_TREE_UPDATER_APPROX_H_

#include <limits>
#include <utility>
#include <vector>
#include <algorithm>
#include <unordered_map>

#include "../common/partition_builder.h"
#include "../common/opt_partition_builder.h"
#include "../common/column_matrix.h"
#include "../common/random.h"
#include "constraints.h"
#include "driver.h"
#include "hist/evaluate_splits.h"
#include "hist/expand_entry.h"
#include "hist/param.h"
#include "param.h"
#include "xgboost/json.h"
#include "xgboost/tree_updater.h"

namespace xgboost {
namespace tree {
class ApproxRowPartitioner {
  common::OptPartitionBuilder opt_partition_builder_;
  std::vector<uint16_t> node_ids_;

 public:
  bst_row_t base_rowid = 0;
  bool is_loss_guided = false;

  static auto SearchCutValue(bst_row_t ridx, bst_feature_t fidx, GHistIndexMatrix const &index,
                             std::vector<uint32_t> const &cut_ptrs,
                             std::vector<float> const &cut_values) {
    int32_t gidx = -1;
// <<<<<<< HEAD
// =======
    auto const &row_ptr = index.row_ptr;
    // CHECK_LT(ridx, row_ptr.size());
    auto get_rid = [&](size_t ridx) { return row_ptr[ridx]; };

// >>>>>>> a20b4d1a... partition optimizations
    if (index.IsDense()) {
      // RowIdx returns the starting pos of this row
      gidx = index.index[get_rid(ridx) + fidx];
    } else {
      auto begin = get_rid(ridx);
      auto end = get_rid(ridx + 1);
      auto f_begin = cut_ptrs[fidx];
      auto f_end = cut_ptrs[fidx + 1];
      gidx = common::BinarySearchBin(begin, end, index.index, f_begin, f_end);
    }
    if (gidx == -1) {
      return std::numeric_limits<float>::quiet_NaN();
    }
    return cut_values[gidx];
  }

 public:
  void UpdatePosition(GenericParameter const *ctx,
                      const GHistIndexMatrix & index,
                      std::unordered_map<uint32_t, CPUExpandEntry>* candidates,
                      RegTree const *p_tree,
                      std::unordered_map<uint32_t, bool>* smalest_nodes_mask,
                      std::unordered_map<uint32_t, uint16_t>* curr_level_nodes,
                      const std::vector<uint16_t>& complete_trees_depth_wise,
                      int depth, bool is_left_small = true) {
    auto const &cut_values = index.cut.Values();
    auto const &cut_ptrs = index.cut.Ptrs();
    const size_t depth_begin = opt_partition_builder_.DepthBegin(complete_trees_depth_wise,
                                                               p_tree, is_loss_guided, depth);
    const size_t depth_size = opt_partition_builder_.DepthSize(index, complete_trees_depth_wise,
                                                             p_tree, is_loss_guided, depth);
    auto node_ptr = p_tree->GetCategoriesMatrix().node_ptr;
    auto categories = p_tree->GetCategoriesMatrix().categories;

    std::vector<uint16_t> curr_level_nodes_data_vec;
    std::vector<bool> smalest_nodes_mask_vec;
    if (opt_partition_builder_.max_depth != 0 && false) {
      curr_level_nodes_data_vec.resize((1 << (opt_partition_builder_.max_depth + 2)), 0);
      smalest_nodes_mask_vec.resize((1 << (opt_partition_builder_.max_depth + 2)), 0);
      for (size_t nid = 0; nid < (1 << (opt_partition_builder_.max_depth + 2)); ++nid) {
        curr_level_nodes_data_vec[nid] = (*curr_level_nodes)[nid];
        smalest_nodes_mask_vec[nid] = (*smalest_nodes_mask)[nid];
      }
    #pragma omp parallel num_threads(ctx->Threads())
      {
        size_t tid = omp_get_thread_num();
        size_t chunck_size = common::GetBlockSize(depth_size, ctx->Threads());
        size_t begin = chunck_size * tid;
        size_t end = std::min(begin + chunck_size, depth_size);
        begin += depth_begin;
        end += depth_begin;
        opt_partition_builder_.PartitionRange(tid, begin, end,
                                              node_ids_.data(), [&](size_t row_id) {
            const uint16_t check_node_id = (~(static_cast<uint16_t>(1) << 15)) &
                                          node_ids_.data()[row_id];
            size_t node_in_set = check_node_id;
            // CHECK_LT(node_in_set, candidates.size());
            auto candidate = (*candidates)[node_in_set];
            auto is_cat = candidate.split.is_cat;
            const int32_t nid = candidate.nid;
            // CHECK_EQ(nid, node_in_set);
            auto fidx = candidate.split.SplitIndex();
            auto cut_value = SearchCutValue(row_id, fidx, index, cut_ptrs, cut_values);
            if (std::isnan(cut_value)) {
              return candidate.split.DefaultLeft();
            }
            bst_node_t nidx = candidate.nid;
            auto segment = node_ptr[nidx];
            auto node_cats = categories.subspan(segment.beg, segment.size);
            bool go_left = true;
              // CHECK_EQ(1, 0);
            if (is_cat) {
              go_left = common::Decision(node_cats, cut_value, candidate.split.DefaultLeft());
            } else {
              // CHECK_EQ(1, 0);
              go_left = cut_value <= candidate.split.split_value;
            }
            return go_left;
          }, &smalest_nodes_mask_vec, &curr_level_nodes_data_vec, is_loss_guided, depth);
      }
    } else {
    #pragma omp parallel num_threads(ctx->Threads())
      {
        size_t tid = omp_get_thread_num();
        size_t chunck_size = common::GetBlockSize(depth_size, ctx->Threads());
        size_t begin = chunck_size * tid;
        size_t end = std::min(begin + chunck_size, depth_size);
        begin += depth_begin;
        end += depth_begin;
        opt_partition_builder_.PartitionRange(tid, begin, end,
                                              node_ids_.data(), [&](size_t row_id) {
            const uint16_t check_node_id = (~(static_cast<uint16_t>(1) << 15)) &
                                          node_ids_.data()[row_id];
            size_t node_in_set = check_node_id;
            // CHECK_LT(node_in_set, candidates.size());
            auto candidate = (*candidates)[node_in_set];
            auto is_cat = candidate.split.is_cat;
            const int32_t nid = candidate.nid;
            // CHECK_EQ(nid, node_in_set);
            auto fidx = candidate.split.SplitIndex();
            auto cut_value = SearchCutValue(row_id, fidx, index, cut_ptrs, cut_values);
            if (std::isnan(cut_value)) {
              return candidate.split.DefaultLeft();
            }
            bst_node_t nidx = candidate.nid;
            auto segment = node_ptr[nidx];
            auto node_cats = categories.subspan(segment.beg, segment.size);
            bool go_left = true;
              // CHECK_EQ(1, 0);
            if (is_cat) {
              go_left = common::Decision(node_cats, cut_value, candidate.split.DefaultLeft());
            } else {
              // CHECK_EQ(1, 0);
              go_left = cut_value <= candidate.split.split_value;
            }
            return go_left;
          }, smalest_nodes_mask, curr_level_nodes, is_loss_guided, depth);
      }
    }
    /*Calculate threads work: UpdateRowBuffer, UpdateThreadsWork*/
    if (depth != opt_partition_builder_.max_depth || is_loss_guided) {
      opt_partition_builder_.UpdateRowBuffer(complete_trees_depth_wise,
                                            p_tree, index, index.cut.Ptrs().size() - 1, depth,
                                            node_ids_, is_loss_guided);
      opt_partition_builder_.UpdateThreadsWork(complete_trees_depth_wise, index,
                                              index.cut.Ptrs().size() - 1, depth, is_loss_guided,
                                              is_left_small, true);
    }
  }

  std::vector<uint16_t> &GetNodeAssignments() { return node_ids_; }

  auto const &GetThreadTasks(const size_t tid) const {
    return opt_partition_builder_.GetSlices(tid);
  }

  auto const &GetOptPartition() const {
    return opt_partition_builder_;
  }

  ApproxRowPartitioner() = default;
  explicit ApproxRowPartitioner(GenericParameter const *ctx,
                                GHistIndexMatrix const &gmat,
                                common::ColumnMatrix const &column_matrix,
                                const RegTree* p_tree_local,
                                size_t max_depth,
                                bool is_loss_guide) {
    is_loss_guided = is_loss_guide;

    const size_t block_size = common::GetBlockSize(gmat.row_ptr.size() - 1, ctx->Threads());

    if (is_loss_guided) {
      opt_partition_builder_.ResizeRowsBuffer(gmat.row_ptr.size() - 1);
      uint32_t* row_set_collection_vec_p = opt_partition_builder_.GetRowsBuffer();
      #pragma omp parallel num_threads(ctx->Threads())
      {
        const size_t tid = omp_get_thread_num();
        const size_t ibegin = tid * block_size;
        const size_t iend = std::min(static_cast<size_t>(ibegin + block_size),
            static_cast<size_t>(gmat.row_ptr.size() - 1));
        for (size_t i = ibegin; i < iend; ++i) {
          row_set_collection_vec_p[i] = i;
        }
      }
    }
    switch (column_matrix.GetTypeSize()) {
      case common::kUint8BinsTypeSize:
        opt_partition_builder_.Init<uint8_t>(gmat, column_matrix, p_tree_local,
                                                ctx->Threads(), max_depth,
                                                is_loss_guide);
        break;
      case common::kUint16BinsTypeSize:
        opt_partition_builder_.Init<uint16_t>(gmat, column_matrix, p_tree_local,
                                                ctx->Threads(), max_depth,
                                                is_loss_guide);
        break;
      case common::kUint32BinsTypeSize:
        opt_partition_builder_.Init<uint32_t>(gmat, column_matrix, p_tree_local,
                                                ctx->Threads(), max_depth,
                                                is_loss_guide);
        break;
      default:
        CHECK(false);  // no default behavior
    }
    opt_partition_builder_.SetSlice(0, 0, gmat.row_ptr.size() - 1);
    node_ids_.resize(gmat.row_ptr.size() - 1, 0);
  }
};
}  // namespace tree
}  // namespace xgboost
#endif  // XGBOOST_TREE_UPDATER_APPROX_H_
