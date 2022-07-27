/*!
 * Copyright 2021-2022 XGBoost contributors
 *
 * \brief Implementation for the approx tree method.
 */
#ifndef XGBOOST_TREE_COMMON_ROW_PARTITIONER_H_
#define XGBOOST_TREE_COMMON_ROW_PARTITIONER_H_

#include <limits>
#include <utility>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <memory>

#include "../common/opt_partition_builder.h"
#include "../common/column_matrix.h"
#include "../common/random.h"
#include "constraints.h"
#include "driver.h"
#include "hist/evaluate_splits.h"
#include "hist/expand_entry.h"
#include "param.h"
#include "xgboost/generic_parameters.h"
#include "xgboost/json.h"
#include "xgboost/tree_updater.h"

namespace xgboost {
namespace tree {
class CommonRowPartitioner {
 public:
  using NodeIdListT = std::vector<uint16_t>;

  const uint32_t one = 1;
  const uint32_t zero = 0;

 private:
  std::unique_ptr<common::Monitor> monitor;
  common::OptPartitionBuilder opt_partition_builder_;
  NodeIdListT node_ids_;

 public:
  bst_row_t base_rowid = 0;
  bool is_loss_guided = false;

 private:
  /**
   * \brief Class for storing UpdatePosition template parameters' values for dispatching simplification
   */
    class DispatchParameterSet final {
     public:
        DispatchParameterSet(bool has_missing, common::BinTypeSize bin_type_size,
                             bool loss_guide, bool has_categorical) :
                    has_missing_(has_missing), bin_type_size_(bin_type_size),
                    loss_guide_(loss_guide), has_categorical_(has_categorical) {}

        common::BinTypeSize GetBinTypeSize() const { return bin_type_size_; }

        bool GetHasMissing() const { return has_missing_; }
        bool GetLossGuide() const { return loss_guide_; }
        bool GetHasCategorical() const { return has_categorical_; }

     private:
        bool has_missing_;
        common::BinTypeSize bin_type_size_;
        bool loss_guide_;
        bool has_categorical_;
    };

  /**
   * \brief Class for storing UpdatePosition call params' values
   *   for simplification of dispatching by template parameters
   */
  class UpdatePositionHelper final {
   public:
    UpdatePositionHelper(xgboost::tree::CommonRowPartitioner* row_partitioner,
      GenericParameter const* ctx, GHistIndexMatrix const& gmat,
      std::vector<xgboost::tree::CPUExpandEntry> const& nodes,
      RegTree const* p_tree,
      int depth,
      common::FlexibleContainer<common::SplitNode>* split_info,
      const size_t max_depth,
      NodeIdListT* child_node_ids,
      bool is_left_small = true,
      bool check_is_left_small = false) :
        row_partitioner_(*row_partitioner),
        ctx_(ctx),
        gmat_(gmat),
        nodes_(nodes),
        p_tree_(p_tree),
        depth_(depth),
        split_info_(split_info),
        max_depth_(max_depth),
        child_node_ids_(child_node_ids),
        is_left_small_(is_left_small),
        check_is_left_small_(check_is_left_small) { }

    template <bool missing, typename BinType, bool is_loss_guide, bool has_cat>
    void Call() {
      row_partitioner_.template UpdatePosition<missing, BinType, is_loss_guide, has_cat>(
        ctx_,
        gmat_,
        nodes_,
        p_tree_,
        depth_,
        split_info_,
        max_depth_,
        child_node_ids_,
        is_left_small_,
        check_is_left_small_);
    }

   private:
    xgboost::tree::CommonRowPartitioner& row_partitioner_;
    GenericParameter const* ctx_;
    GHistIndexMatrix const& gmat_;
    std::vector<xgboost::tree::CPUExpandEntry> const& nodes_;
    RegTree const* p_tree_;
    int depth_;
    common::FlexibleContainer<common::SplitNode>* split_info_;
    const size_t max_depth_;
    NodeIdListT* child_node_ids_;
    bool is_left_small_;
    bool check_is_left_small_;
  };

 public:
  // clang-format off
  template <typename Type>
  void InitilizerCall(Type) {}

  template <bool ... switch_values_set>
  void DispatchFromHasMissing(UpdatePositionHelper&& pos_updater,
    const DispatchParameterSet&& dispatch_values,
    std::integer_sequence<bool, switch_values_set...>) {
    InitilizerCall<std::initializer_list<uint32_t>>({(dispatch_values.GetHasMissing()
      == switch_values_set ?
      DispatchFromBinType<switch_values_set>(std::move(pos_updater), std::move(dispatch_values),
      std::move(common::BinTypeSizeSequence{})), one : zero)...});
  }

  template <bool missing, uint32_t ... switch_values_set>
  void DispatchFromBinType(UpdatePositionHelper&& pos_updater,
    const DispatchParameterSet&& dispatch_values,
                           std::integer_sequence<uint32_t, switch_values_set...>) {
      InitilizerCall<std::initializer_list<uint32_t>>({(dispatch_values.GetBinTypeSize() ==
        switch_values_set ?
        DispatchFromLossGuide<missing,
        typename common::BinTypeMap<switch_values_set>::Type>(std::move(pos_updater),
        std::move(dispatch_values), std::move(common::BoolSequence{})), one : zero)...});
  }

  template <bool missing, typename BinType, bool ... switch_values_set>
  void DispatchFromLossGuide(UpdatePositionHelper&& pos_updater,
                             const DispatchParameterSet&& dispatch_values,
                             std::integer_sequence<bool, switch_values_set...>) {
    InitilizerCall<std::initializer_list<uint32_t>>({
      (dispatch_values.GetLossGuide() == switch_values_set ?
      DispatchFromHasCategorical<missing, BinType,
      switch_values_set>(std::move(pos_updater),
      std::move(dispatch_values),
      std::move(common::BoolSequence{})), one : zero)...});
  }

  template <bool missing, typename BinType, bool is_loss_guide, bool ... switch_values_set>
  void DispatchFromHasCategorical(UpdatePositionHelper&& pos_updater,
                                  const DispatchParameterSet&& dispatch_values,
                                  std::integer_sequence<bool, switch_values_set...>) {
    InitilizerCall<std::initializer_list<uint32_t>>({
      (dispatch_values.GetHasCategorical() == switch_values_set ?
      pos_updater.template Call<missing, BinType, is_loss_guide, switch_values_set>(), one
      : zero)...});
  }

  template <typename ... Args>
  void UpdatePositionDispatched(DispatchParameterSet&& dispatch_params, Args&& ... args) {
      UpdatePositionHelper helper(this, std::forward<Args>(args)...);
      DispatchFromHasMissing(std::move(helper), std::move(dispatch_params),
      std::move(common::BoolSequence{}));
  }
  // clang-format on

  /**
   * \brief Turn split values into discrete bin indices.
   */
  void FindSplitConditions(const std::vector<CPUExpandEntry> &nodes,
                           const RegTree &tree, const GHistIndexMatrix &gmat,
                           common::FlexibleContainer<common::SplitNode>* split_info) {
    for (const auto& node : nodes) {
      const int32_t nid = node.nid;
      const bst_uint fid = tree[nid].SplitIndex();
      const bst_float split_pt = tree[nid].SplitCond();
      const uint32_t lower_bound = gmat.cut.Ptrs()[fid];
      const uint32_t upper_bound = gmat.cut.Ptrs()[fid + 1];
      int32_t split_cond = -1;
      // convert floating-point split_pt into corresponding bin_id
      // split_cond = -1 indicates that split_pt is less than all known cut points
      CHECK_LT(upper_bound, static_cast<uint32_t>(std::numeric_limits<int32_t>::max()));
      for (uint32_t bound = lower_bound; bound < upper_bound; ++bound) {
        if (split_pt == gmat.cut.Values()[bound]) {
          split_cond = static_cast<int32_t>(bound);
        }
      }
      (*split_info)[nid].condition = split_cond;
    }
  }

  template <bool any_cat>
  auto GetPredicate(const RegTree& tree, const GHistIndexMatrix& gmat) {
    auto pred = [&tree, &gmat](auto ridx, auto bin_id, auto nid, auto split_cond) {
      if (!any_cat) {
        return bin_id <= split_cond;
      }
      bool is_cat = tree.GetSplitTypes()[nid] == FeatureType::kCategorical;
      if (any_cat && is_cat) {
        auto const& index = gmat.index;
        auto const& cut_values = gmat.cut.Values();
        auto const& cut_ptrs = gmat.cut.Ptrs();

        const bst_uint fid = tree[nid].SplitIndex();
        const bool default_left = tree[nid].DefaultLeft();
        auto node_cats = tree.NodeCats(nid);
        auto begin = gmat.RowIdx(ridx);
        auto end = gmat.RowIdx(ridx + 1);
        auto f_begin = cut_ptrs[fid];
        auto f_end = cut_ptrs[fid + 1];
        // bypassing the column matrix as we need the cut value instead of bin idx for categorical
        // features.
        auto gidx = BinarySearchBin(begin, end, index, f_begin, f_end);
        bool go_left;
        if (gidx == -1) {
          go_left = default_left;
        } else {
          go_left = Decision(node_cats, cut_values[gidx], default_left);
        }
        return go_left;
      } else {
        return bin_id <= split_cond;
      }
    };
    return pred;
  }

  template <bool any_missing, typename BinIdxType,
            bool is_loss_guided, bool any_cat>
  void UpdatePosition(GenericParameter const* ctx, GHistIndexMatrix const& gmat,
    std::vector<CPUExpandEntry> const& nodes, RegTree const* p_tree,
    int depth,
    common::FlexibleContainer<common::SplitNode>* split_info,
    const size_t max_depth,
    NodeIdListT* child_node_ids,
    bool is_left_small = true,
    bool check_is_left_small = false) {
    opt_partition_builder_.SetDepth(depth);
    opt_partition_builder_.SetNodeIdsPtr(node_ids_.data());
    common::ColumnMatrix const& column_matrix = gmat.Transpose();
    if (column_matrix.GetIndexData() != opt_partition_builder_.data_hash ||
        column_matrix.GetMissing() != opt_partition_builder_.missing_ptr ||
        column_matrix.GetRowId() != opt_partition_builder_.row_ind_ptr) {
        opt_partition_builder_.Init(column_matrix, gmat, p_tree,
                                    ctx->Threads(), max_depth,
                                    is_loss_guided);
    }

    // 1. Find split condition for each split
    const size_t n_nodes = nodes.size();
    FindSplitConditions(nodes, *p_tree, gmat, split_info);
    // 2.1 Create a blocked space of size SUM(samples in each node)
    const uint32_t* offsets = gmat.index.Offset();
    const uint64_t rows_offset = gmat.row_ptr.size() - 1;
    opt_partition_builder_.ResizeSplitNodeIfSmaller(n_nodes);
    uint32_t* split_nodes = opt_partition_builder_.GetSplitNodesPtr();
    for (size_t i = 0; i < n_nodes; ++i) {
        const int32_t nid = nodes[i].nid;
        split_nodes[i] = nid;
        const uint64_t fid = (*p_tree)[nid].SplitIndex();
        (*split_info)[nid].ind = fid*((gmat.IsDense() ? rows_offset : 1));
        (*split_info)[nid].condition -= gmat.cut.Ptrs()[fid];
    }

    auto pred = GetPredicate<any_cat>(*p_tree, gmat);

    const size_t n_features = gmat.cut.Ptrs().size() - 1;
    int nthreads = std::max(ctx->Threads(), 1);
    const size_t depth_begin = opt_partition_builder_.template DepthBegin<is_loss_guided>(
                                                                           *child_node_ids);
    const size_t depth_size = opt_partition_builder_.template DepthSize<is_loss_guided>(
                                                                   gmat, *child_node_ids);

    monitor->Start("CommonPartition");
    size_t chunck_size = common::GetBlockSize(depth_size, nthreads);
    size_t thread_size = chunck_size;
    #pragma omp parallel num_threads(nthreads)
    {
      size_t tid = omp_get_thread_num();
      const BinIdxType* numa = tid < nthreads/2 ?
      reinterpret_cast<const BinIdxType*>(column_matrix.GetIndexData()) :
      reinterpret_cast<const BinIdxType*>(column_matrix.GetIndexData());

      size_t begin = thread_size * tid;
      size_t end = std::min(begin + thread_size, depth_size);
      begin += depth_begin;
      end += depth_begin;
      common::RowIndicesRange range{begin, end};
      opt_partition_builder_.template CommonPartition
          <BinIdxType, is_loss_guided, !any_missing, any_cat>
          (column_matrix, pred, numa, tid, range, *split_info);
    }
    monitor->Stop("CommonPartition");

    if (depth != max_depth || is_loss_guided) {
        monitor->Start("UpdateRowBuffer");
        opt_partition_builder_.template UpdateRowBuffer<is_loss_guided>(
                                              *child_node_ids, gmat,
                                              n_features);
        monitor->Stop("UpdateRowBuffer");
        monitor->Start("UpdateThreadsWork");
        opt_partition_builder_.template UpdateThreadsWork<is_loss_guided>(
                                                *child_node_ids, gmat,
                                                n_features,
                                                is_left_small,
                                                check_is_left_small);
        monitor->Stop("UpdateThreadsWork");
  }
}

  NodeIdListT &GetNodeAssignments() { return node_ids_; }

  void LeafPartition(Context const *ctx, RegTree const &tree,
                     std::vector<bst_node_t> *p_out_position) const {
    auto& h_pos = *p_out_position;
    const uint16_t* node_ids_data_ptr = node_ids_.data();
    h_pos.resize(node_ids_.size(), std::numeric_limits<bst_node_t>::max());
    int nthreads = 1;
    xgboost::common::ParallelFor(node_ids_.size(), nthreads, [&](size_t i) {
      h_pos[i] = node_ids_data_ptr[i];
    });
  }

  auto const &GetThreadTasks(const size_t tid) const {
    return opt_partition_builder_.GetSlices(tid);
  }

  auto const &GetOptPartition() const {
    return opt_partition_builder_;
  }

  // CommonRowPartitioner() = default;
  CommonRowPartitioner() {
    monitor = std::make_unique<common::Monitor>();
    monitor->Init("CommonRowPartitioner");
  }

  explicit CommonRowPartitioner(GenericParameter const *ctx,
                                GHistIndexMatrix const &gmat,
                                const RegTree* p_tree_local,
                                size_t max_depth,
                                bool is_loss_guide) {
    monitor = std::make_unique<common::Monitor>();
    monitor->Init("CommonRowPartitioner");
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
    common::ColumnMatrix const &column_matrix = gmat.Transpose();
    opt_partition_builder_.Init(column_matrix, gmat, p_tree_local,
                                            ctx->Threads(), max_depth,
                                            is_loss_guide);
    opt_partition_builder_.SetSlice(0, 0, gmat.row_ptr.size() - 1);
    node_ids_.resize(gmat.row_ptr.size() - 1, 0);
  }

  void Reset(GenericParameter const *ctx,
                     GHistIndexMatrix const &gmat,
                     const RegTree* p_tree_local,
                     size_t max_depth,
                     bool is_loss_guide) {
    common::ColumnMatrix const & column_matrix = gmat.Transpose();
    const size_t block_size = common::GetBlockSize(gmat.row_ptr.size() - 1, ctx->Threads());

    if (is_loss_guide) {
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
    opt_partition_builder_.Init(column_matrix, gmat, p_tree_local,
                                            ctx->Threads(), max_depth,
                                            is_loss_guide);
    opt_partition_builder_.SetSlice(0, 0, gmat.row_ptr.size() - 1);
    node_ids_.resize(gmat.row_ptr.size() - 1, 0);
  }
};
}  // namespace tree
}  // namespace xgboost
#endif  // XGBOOST_TREE_COMMON_ROW_PARTITIONER_H_