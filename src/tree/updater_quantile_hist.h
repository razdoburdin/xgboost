/*!
 * Copyright 2017-2022 by XGBoost Contributors
 * \file updater_quantile_hist.h
 * \brief use quantized feature values to construct a tree
 * \author Philip Cho, Tianqi Chen, Egor Smirnov
 */
#ifndef XGBOOST_TREE_UPDATER_QUANTILE_HIST_H_
#define XGBOOST_TREE_UPDATER_QUANTILE_HIST_H_

#include <dmlc/timer.h>
#include <rabit/rabit.h>
#include <xgboost/tree_updater.h>

#include <algorithm>
#include <unordered_map>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "xgboost/data.h"
#include "xgboost/json.h"

#include "hist/evaluate_splits.h"
#include "hist/histogram.h"
#include "hist/expand_entry.h"
#include "hist/param.h"

#include "constraints.h"
#include "./param.h"
#include "./driver.h"
#include "./split_evaluator.h"
#include "../common/random.h"
#include "../common/timer.h"
#include "../common/hist_util.h"
#include "../common/row_set.h"
#include "../common/opt_partition_builder.h"
#include "../common/partition_builder.h"
#include "../common/column_matrix.h"

namespace xgboost {
struct RandomReplace {
 public:
  // similar value as for minstd_rand
  static constexpr uint64_t kBase = 16807;
  static constexpr uint64_t kMod = static_cast<uint64_t>(1) << 63;

  using EngineT = std::linear_congruential_engine<uint64_t, kBase, 0, kMod>;

  /*
    Right-to-left binary method: https://en.wikipedia.org/wiki/Modular_exponentiation
  */
  static uint64_t SimpleSkip(uint64_t exponent, uint64_t initial_seed,
                             uint64_t base, uint64_t mod) {
    CHECK_LE(exponent, mod);
    uint64_t result = 1;
    while (exponent > 0) {
      if (exponent % 2 == 1) {
        result = (result * base) % mod;
      }
      base = (base * base) % mod;
      exponent = exponent >> 1;
    }
    // with result we can now find the new seed
    return (result * initial_seed) % mod;
  }

  template<typename Condition, typename ContainerData>
  static void MakeIf(Condition condition, const typename ContainerData::value_type replace_value,
                     const uint64_t initial_seed, const size_t ibegin,
                     const size_t iend, ContainerData* gpair) {
    ContainerData& gpair_ref = *gpair;
    const uint64_t displaced_seed = SimpleSkip(ibegin, initial_seed, kBase, kMod);
    EngineT eng(displaced_seed);
    for (size_t i = ibegin; i < iend; ++i) {
      if (condition(i, eng)) {
        gpair_ref[i] = replace_value;
      }
    }
  }
};

namespace tree {
class HistRowPartitioner {
  // heuristically chosen block size of parallel partitioning
  static constexpr size_t kPartitionBlockSize = 2048;
  // worker class that partition a block of rows
  common::PartitionBuilder<kPartitionBlockSize> partition_builder_;
  common::OptPartitionBuilder opt_partition_builder_;
  std::vector<uint16_t> node_ids_;
  // storage for row index
  common::RowSetCollection row_set_collection_;
    common::Monitor builder_monitor_;

  /**
   * \brief Turn split values into discrete bin indices.
   */
  static void FindSplitConditions(const std::vector<CPUExpandEntry>& nodes, const RegTree& tree,
                                  const GHistIndexMatrix& gmat,
                                  std::unordered_map<uint32_t, int32_t>* split_conditions);
  /**
   * \brief Update the row set for new splits specifed by nodes.
   */
  void AddSplitsToRowSet(const std::vector<CPUExpandEntry>& nodes, RegTree const* p_tree);

 public:
  bst_row_t base_rowid = 0;

 public:
  HistRowPartitioner(GenericParameter const *ctx,
                     GHistIndexMatrix const &gmat,
                     common::ColumnMatrix const & column_matrix,
                     const RegTree* p_tree_local,
                     size_t max_depth,
                     bool is_loss_guide) {
    builder_monitor_.Init("HistRowPartitioner");
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

  void Reset(GenericParameter const *ctx,
                     GHistIndexMatrix const &gmat,
                     common::ColumnMatrix const & column_matrix,
                     const RegTree* p_tree_local,
                     size_t max_depth,
                     bool is_loss_guide) {
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

  template <bool any_missing, typename BinIdxType,
            bool is_loss_guided, bool any_cat>
  void UpdatePosition(GenericParameter const* ctx, GHistIndexMatrix const& gmat,
                      common::ColumnMatrix const& column_matrix,
                      std::vector<CPUExpandEntry> const& nodes, RegTree const* p_tree,
                      int depth,
                      std::unordered_map<uint32_t, bool>* smalest_nodes_mask_ptr,
                      const bool loss_guide,
                      std::unordered_map<uint32_t, int32_t>* split_conditions_,
                      std::unordered_map<uint32_t, uint64_t>* split_ind_, const size_t max_depth,
                      std::vector<uint16_t>* complete_trees_depth_wise_,
                      std::unordered_map<uint32_t, uint16_t>* curr_level_nodes_) {
    // 1. Find split condition for each split
    const size_t n_nodes = nodes.size();
    FindSplitConditions(nodes, *p_tree, gmat, split_conditions_);
    // 2.1 Create a blocked space of size SUM(samples in each node)
    const uint32_t* offsets = gmat.index.Offset();
    const uint64_t rows_offset = gmat.row_ptr.size() - 1;
    std::vector<uint32_t> split_nodes(n_nodes, 0);
    for (size_t i = 0; i < n_nodes; ++i) {
        const int32_t nid = nodes[i].nid;
        split_nodes[i] = nid;
        const uint64_t fid = (*p_tree)[nid].SplitIndex();
        (*split_ind_)[nid] = fid*((gmat.IsDense() ? rows_offset : 1));
        (*split_conditions_)[nid] = (*split_conditions_)[nid] - gmat.cut.Ptrs()[fid];
    }
    std::vector<uint16_t> curr_level_nodes_data_vec;
    std::vector<uint64_t> split_ind_data_vec;
    std::vector<int32_t> split_conditions_data_vec;
    std::vector<bool> smalest_nodes_mask_vec;
    if (max_depth != 0) {
      curr_level_nodes_data_vec.resize((1 << (max_depth + 2)), 0);
      split_ind_data_vec.resize((1 << (max_depth + 2)), 0);
      split_conditions_data_vec.resize((1 << (max_depth + 2)), 0);
      smalest_nodes_mask_vec.resize((1 << (max_depth + 2)), 0);
      for (size_t nid = 0; nid < (1 << (max_depth + 2)); ++nid) {
        curr_level_nodes_data_vec[nid] = (*curr_level_nodes_)[nid];
        split_ind_data_vec[nid] = (*split_ind_)[nid];
        split_conditions_data_vec[nid] = (*split_conditions_)[nid];
        smalest_nodes_mask_vec[nid] = (*smalest_nodes_mask_ptr)[nid];
      }
    }
    const size_t n_features = gmat.cut.Ptrs().size() - 1;
    int nthreads = ctx->Threads();
    // nthreads = std::min(nthreads, omp_get_max_threads());
    nthreads = std::max(nthreads, 1);
    const size_t depth_begin = opt_partition_builder_.DepthBegin(*complete_trees_depth_wise_,
                                                                p_tree, loss_guide, depth);
    const size_t depth_size = opt_partition_builder_.DepthSize(gmat, *complete_trees_depth_wise_,
                                                              p_tree, loss_guide, depth);

    auto const& index = gmat.index;
    auto const& cut_values = gmat.cut.Values();
    auto const& cut_ptrs = gmat.cut.Ptrs();
    RegTree const tree = *p_tree;
    auto pred = [&](auto ridx, auto bin_id, auto nid, auto split_cond) {
      if (!any_cat) {
        return bin_id <= split_cond;
      }
      bool is_cat = tree.GetSplitTypes()[nid] == FeatureType::kCategorical;
      if (any_cat && is_cat) {
        const bst_uint fid = tree[nid].SplitIndex();
        const bool default_left = tree[nid].DefaultLeft();
        const auto column_ptr = column_matrix.GetColumn<BinIdxType, any_missing>(fid);
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
    builder_monitor_.Start("CommonPartition");
    if (max_depth != 0) {
    #pragma omp parallel num_threads(nthreads)
      {
        size_t tid = omp_get_thread_num();
        const BinIdxType* numa = tid < nthreads/2 ?
          reinterpret_cast<const BinIdxType*>(column_matrix.GetIndexData()) :
          reinterpret_cast<const BinIdxType*>(column_matrix.GetIndexData());
          // reinterpret_cast<const BinIdxType*>(column_matrix.GetIndexSecondData());
        size_t chunck_size = common::GetBlockSize(depth_size, nthreads);
        size_t thread_size = chunck_size;
        size_t begin = thread_size * tid;
        size_t end = std::min(begin + thread_size, depth_size);
        begin += depth_begin;
        end += depth_begin;
        opt_partition_builder_.template CommonPartition<BinIdxType,
                                                        is_loss_guided,
                                                        !any_missing,
                                                        any_cat>(tid, begin, end, numa,
                                                                node_ids_.data(),
                                                                &split_conditions_data_vec,
                                                                &split_ind_data_vec,
                                                                &smalest_nodes_mask_vec,
                                                                &curr_level_nodes_data_vec,
                                                                column_matrix,
                                                                split_nodes, pred, depth);
      }
    } else {
    #pragma omp parallel num_threads(nthreads)
      {
        size_t tid = omp_get_thread_num();
        const BinIdxType* numa = tid < nthreads/2 ?
          reinterpret_cast<const BinIdxType*>(column_matrix.GetIndexData()) :
          reinterpret_cast<const BinIdxType*>(column_matrix.GetIndexData());
          // reinterpret_cast<const BinIdxType*>(column_matrix.GetIndexSecondData());
        size_t chunck_size = common::GetBlockSize(depth_size, nthreads);
        size_t thread_size = chunck_size;
        size_t begin = thread_size * tid;
        size_t end = std::min(begin + thread_size, depth_size);
        begin += depth_begin;
        end += depth_begin;
        opt_partition_builder_.template CommonPartition<BinIdxType,
                                                        is_loss_guided,
                                                        !any_missing,
                                                        any_cat>(tid, begin, end, numa,
                                                                 node_ids_.data(),
                                                                 split_conditions_,
                                                                 split_ind_,
                                                                 smalest_nodes_mask_ptr,
                                                                 curr_level_nodes_,
                                                                 column_matrix,
                                                                 split_nodes, pred, depth);
      }
    }
    builder_monitor_.Stop("CommonPartition");

    if (depth != max_depth || loss_guide) {
      builder_monitor_.Start("UpdateRowBuffer&UpdateThreadsWork");
      opt_partition_builder_.UpdateRowBuffer(*complete_trees_depth_wise_,
                                            p_tree, gmat, n_features, depth,
                                            node_ids_, is_loss_guided);
      opt_partition_builder_.UpdateThreadsWork(*complete_trees_depth_wise_, gmat,
                                              n_features, depth, is_loss_guided);
      builder_monitor_.Stop("UpdateRowBuffer&UpdateThreadsWork");
    }
}
  std::vector<uint16_t> &GetNodeAssignments() { return node_ids_; }

  auto const &GetThreadTasks(const size_t tid) const {
    return opt_partition_builder_.GetSlices(tid);
  }

  auto const &GetOptPartition() const {
    return opt_partition_builder_;
  }
};

inline BatchParam HistBatch(TrainParam const& param) {
  return {param.max_bin, param.sparse_threshold};
}

/*! \brief construct a tree using quantized feature values */
class QuantileHistMaker: public TreeUpdater {
 public:
  explicit QuantileHistMaker(ObjInfo task) : task_{task} {
    updater_monitor_.Init("QuantileHistMaker");
  }
  void Configure(const Args& args) override;

  void Update(HostDeviceVector<GradientPair>* gpair,
              DMatrix* dmat,
              const std::vector<RegTree*>& trees) override;

  bool UpdatePredictionCache(const DMatrix *data,
                             linalg::VectorView<float> out_preds) override;

  void LoadConfig(Json const& in) override {
    auto const& config = get<Object const>(in);
    FromJson(config.at("train_param"), &this->param_);
    try {
      FromJson(config.at("cpu_hist_train_param"), &this->hist_maker_param_);
    } catch (std::out_of_range&) {
      // XGBoost model is from 1.1.x, so 'cpu_hist_train_param' is missing.
      // We add this compatibility check because it's just recently that we (developers) began
      // persuade R users away from using saveRDS() for model serialization. Hopefully, one day,
      // everyone will be using xgb.save().
      LOG(WARNING)
        << "Attempted to load internal configuration for a model file that was generated "
        << "by a previous version of XGBoost. A likely cause for this warning is that the model "
        << "was saved with saveRDS() in R or pickle.dump() in Python. We strongly ADVISE AGAINST "
        << "using saveRDS() or pickle.dump() so that the model remains accessible in current and "
        << "upcoming XGBoost releases. Please use xgb.save() instead to preserve models for the "
        << "long term. For more details and explanation, see "
        << "https://xgboost.readthedocs.io/en/latest/tutorials/saving_model.html";
      this->hist_maker_param_.UpdateAllowUnknown(Args{});
    }
  }
  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["train_param"] = ToJson(param_);
    out["cpu_hist_train_param"] = ToJson(hist_maker_param_);
  }

  char const* Name() const override {
    return "grow_quantile_histmaker";
  }

 protected:
  CPUHistMakerTrainParam hist_maker_param_;
  // training parameter
  TrainParam param_;
  // column accessor
  common::ColumnMatrix column_matrix_;
  DMatrix const* p_last_dmat_ {nullptr};
  bool is_gmat_initialized_ {false};

  // actual builder that runs the algorithm
  template<typename GradientSumT>
  struct Builder {
   public:
    using GradientPairT = xgboost::detail::GradientPairInternal<GradientSumT>;
    // constructor
    explicit Builder(const size_t n_trees, const TrainParam& param,
                     std::unique_ptr<TreeUpdater> pruner, DMatrix const* fmat, ObjInfo task,
                     GenericParameter const* ctx)
        : n_trees_(n_trees),
          param_(param),
          pruner_(std::move(pruner)),
          p_last_fmat_(fmat),
          histogram_builder_{new HistogramBuilder<GradientSumT, CPUExpandEntry>},
          task_{task},
          ctx_{ctx} {
      builder_monitor_.Init("Quantile::Builder");
    }
    // update one tree, growing
    void Update(const GHistIndexMatrix& gmat, const common::ColumnMatrix& column_matrix,
                HostDeviceVector<GradientPair>* gpair, DMatrix* p_fmat, RegTree* p_tree);

    bool UpdatePredictionCache(const DMatrix* data,
                               linalg::VectorView<float> out_preds);

   protected:
    // initialize temp data structure
    template <typename BinIdxType>
    void InitData(const GHistIndexMatrix& gmat,
                  const common::ColumnMatrix& column_matrix,
                  const DMatrix& fmat,
                  const RegTree& tree,
                  std::vector<GradientPair>* gpair);

    size_t GetNumberOfTrees();

    void InitSampling(const DMatrix& fmat, std::vector<GradientPair>* gpair);
    void FindSplitConditions(const std::vector<CPUExpandEntry>& nodes, const RegTree& tree,
                             const GHistIndexMatrix& gmat,
                             std::unordered_map<uint32_t, int32_t>* split_conditions);

    template <typename BinIdxType, bool any_missing>
    void InitRoot(const GHistIndexMatrix &gmat,
                  DMatrix* p_fmat,
                  RegTree *p_tree,
                  const std::vector<GradientPair> &gpair_h,
                  int *num_leaves, std::vector<CPUExpandEntry> *expand);

    // Split nodes to 2 sets depending on amount of rows in each node
    // Histograms for small nodes will be built explicitly
    // Histograms for big nodes will be built by 'Subtraction Trick'
    void SplitSiblings(const std::vector<CPUExpandEntry>& nodes,
                       std::vector<CPUExpandEntry>* nodes_to_evaluate,
                       RegTree *p_tree);

    void AddSplitsToTree(const std::vector<CPUExpandEntry>& expand,
                         RegTree *p_tree,
                         int *num_leaves,
                         std::vector<CPUExpandEntry>* nodes_for_apply_split,
                         std::unordered_map<uint32_t, bool>* smalest_nodes_mask_ptr, size_t depth);

    template <typename BinIdxType, bool any_missing>
    void ExpandTree(const GHistIndexMatrix& gmat,
                    const common::ColumnMatrix& column_matrix,
                    DMatrix* p_fmat,
                    RegTree* p_tree,
                    const std::vector<GradientPair>& gpair_h);

    //  --data fields--
    const size_t n_trees_;
    const TrainParam& param_;
    std::shared_ptr<common::ColumnSampler> column_sampler_{
        std::make_shared<common::ColumnSampler>()};

// <<<<<<< HEAD
// =======
    // the internal row sets
    common::RowSetCollection row_set_collection_;
    std::vector<uint16_t> node_ids_;
    std::unordered_map<uint32_t, uint16_t> curr_level_nodes_;
    std::unordered_map<uint32_t, int32_t> split_conditions_;
    std::unordered_map<uint32_t, uint64_t> split_ind_;
    std::vector<uint16_t> complete_trees_depth_wise_;
// >>>>>>> a20b4d1a... partition optimizations
    std::vector<GradientPair> gpair_local_;

    /*! \brief feature with least # of bins. to be used for dense specialization
               of InitNewNode() */
    uint32_t fid_least_bins_;

    std::unique_ptr<TreeUpdater> pruner_;
    std::unique_ptr<HistEvaluator<GradientSumT, CPUExpandEntry>> evaluator_;
// <<<<<<< HEAD
//     // Right now there's only 1 partitioner in this vector, when external memory is fully
//     // supported we will have number of partitioners equal to number of pages.
    std::vector<HistRowPartitioner> partitioner_;
// =======

    common::OptPartitionBuilder opt_partition_builder_;
// >>>>>>> a20b4d1a... partition optimizations

    // back pointers to tree and data matrix
    const RegTree* p_last_tree_{nullptr};
    DMatrix const* const p_last_fmat_;
    DMatrix* p_last_fmat_mutable_;

    // key is the node id which should be calculated by Subtraction Trick, value is the node which
    // provides the evidence for subtraction
    std::vector<CPUExpandEntry> nodes_for_subtraction_trick_;
    // list of nodes whose histograms would be built explicitly.
    std::vector<CPUExpandEntry> nodes_for_explicit_hist_build_;

    enum class DataLayout { kDenseDataZeroBased, kDenseDataOneBased, kSparseData };
    DataLayout data_layout_;
    std::unique_ptr<HistogramBuilder<GradientSumT, CPUExpandEntry>> histogram_builder_;
    ObjInfo task_;
    // Context for number of threads
    GenericParameter const* ctx_;

    common::Monitor builder_monitor_;
    bool partition_is_initiated_{false};
  };
  common::Monitor updater_monitor_;

  template<typename GradientSumT>
  void SetBuilder(const size_t n_trees, std::unique_ptr<Builder<GradientSumT>>*, DMatrix *dmat);

  template<typename GradientSumT>
  void CallBuilderUpdate(const std::unique_ptr<Builder<GradientSumT>>& builder,
                         HostDeviceVector<GradientPair> *gpair,
                         DMatrix *dmat,
                         GHistIndexMatrix const& gmat,
                         const std::vector<RegTree *> &trees);

 protected:
  std::unique_ptr<Builder<float>> float_builder_;
  std::unique_ptr<Builder<double>> double_builder_;

  std::unique_ptr<TreeUpdater> pruner_;
  ObjInfo task_;
};
}  // namespace tree
}  // namespace xgboost

#endif  // XGBOOST_TREE_UPDATER_QUANTILE_HIST_H_
