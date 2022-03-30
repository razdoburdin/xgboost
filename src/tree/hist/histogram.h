/*!
 * Copyright 2021 by XGBoost Contributors
 */
#ifndef XGBOOST_TREE_HIST_HISTOGRAM_H_
#define XGBOOST_TREE_HIST_HISTOGRAM_H_

#include <algorithm>
#include <limits>
#include <vector>
#include <memory>

#include "rabit/rabit.h"
#include "xgboost/tree_model.h"
#include "../../common/hist_util.h"
#include "../../data/gradient_index.h"
#include "../../common/hist_builder.h"
#include "../../common/opt_partition_builder.h"
#include "../../common/random.h"

namespace xgboost {
namespace tree {
template <typename GradientSumT, typename ExpandEntry> class HistogramBuilder {
  using GradientPairT = xgboost::detail::GradientPairInternal<GradientSumT>;
  using GHistRowT = common::GHistRow<GradientSumT>;

  /*! \brief culmulative histogram of gradients. */
  common::HistCollection<GradientSumT> hist_;
  /*! \brief culmulative local parent histogram of gradients. */
  common::HistCollection<GradientSumT> hist_local_worker_;
  common::ParallelGHistBuilder<GradientSumT> buffer_;
  rabit::Reducer<GradientPairT, GradientPairT::Reduce> reducer_;
  BatchParam param_;
  int32_t n_threads_ = -1;
  size_t total_bins_ = 0;
  float colsample_bytree_ {1.0};
  float colsample_bylevel_ {1.0};
  float colsample_bynode_ {1.0};
  std::shared_ptr<common::ColumnSampler> column_sampler_;
  size_t n_batches_{0};
  // Whether XGBoost is running in distributed environment.
  bool is_distributed_{false};

 public:
  /**
   * \param total_bins       Total number of bins across all features
   * \param max_bin_per_feat Maximum number of bins per feature, same as the `max_bin`
   *                         training parameter.
   * \param n_threads        Number of threads.
   * \param is_distributed   Mostly used for testing to allow injecting parameters instead
   *                         of using global rabit variable.
   */

  void Reset(uint32_t total_bins, int32_t max_bin_per_feat, int32_t n_threads, size_t n_batches,
             int32_t max_depth,
             float colsample_bytree, float colsample_bylevel, float colsample_bynode,
             std::shared_ptr<common::ColumnSampler> column_sampler,
             bool is_distributed = rabit::IsDistributed()) {
    CHECK_GE(n_threads, 1);
    n_threads_ = n_threads;
    n_batches_ = n_batches;
    hist_.Init(total_bins);
    hist_local_worker_.Init(total_bins);
    buffer_.Init(total_bins);
    buffer_.AllocateHistBufer(max_depth, n_threads);
    is_distributed_ = is_distributed;
    colsample_bytree_ = colsample_bytree;
    colsample_bylevel_ = colsample_bylevel;
    colsample_bynode_ = colsample_bynode;
    column_sampler_ = column_sampler;
    total_bins_ = total_bins;
  }

  template <typename BinIdxType, bool read_by_column, bool feature_blocking, bool is_root, bool any_missing, bool column_sampling>
  void BuildLocalHistogramsForSlice(const std::vector<GradientPair> &gpair_h,
                                    GHistIndexMatrix const &gidx, 
                                    const common::Slice& slice, int depth, std::vector<uint16_t>& node_ids,
                                    size_t tid, const common::ColumnMatrix& column_matrix, const std::vector<int>& fids) {
    const size_t n_features = gidx.cut.Ptrs().size() - 1;
    const BinIdxType* numa = gidx.index.data<BinIdxType>();
    common::BuildHist<GradientSumT, BinIdxType, read_by_column, feature_blocking, is_root, any_missing, column_sampling>
                      (gpair_h, slice.addr, slice.b, slice.e, gidx,
                        n_features, numa, node_ids.data(),
                        buffer_.hists_addr[tid], column_matrix,
                        buffer_.local_threads_mapping[tid].data(), fids);
  }

  template <typename BinIdxType, bool any_missing, bool hist_fit_to_l2, bool column_sampling>
  void BuildLocalHistogramsForSlice(const std::vector<GradientPair> &gpair_h,
                                    GHistIndexMatrix const &gidx, 
                                    const common::Slice& slice, int depth, std::vector<uint16_t>& node_ids,
                                    size_t tid, const common::ColumnMatrix& column_matrix, const std::vector<int>& fids) {
    if (depth == 0) {
      constexpr bool feature_blocking = false;
      constexpr bool read_by_column = column_sampling ? true : !hist_fit_to_l2 && !any_missing;
      constexpr bool is_root = true;

      BuildLocalHistogramsForSlice<BinIdxType, read_by_column, feature_blocking, is_root, any_missing, column_sampling>
                                  (gpair_h, gidx, slice, depth, node_ids, tid, column_matrix, fids);
    } else if (depth == 1) {
      constexpr bool feature_blocking = false;
      constexpr bool read_by_column = column_sampling ? true : !hist_fit_to_l2 && !any_missing;
      constexpr bool is_root = false;

      BuildLocalHistogramsForSlice<BinIdxType, read_by_column, feature_blocking, is_root, any_missing, column_sampling>
                                  (gpair_h, gidx, slice, depth, node_ids, tid, column_matrix, fids);
    } else {
      constexpr bool feature_blocking = !hist_fit_to_l2 && !any_missing;
      constexpr bool read_by_column = column_sampling ? true : false;
      constexpr bool is_root = false;

      BuildLocalHistogramsForSlice<BinIdxType, read_by_column, feature_blocking, is_root, any_missing, column_sampling>
                                  (gpair_h, gidx, slice, depth, node_ids, tid, column_matrix, fids);
    }
  }

  template <typename BinIdxType, bool any_missing, bool hist_fit_to_l2, bool column_sampling, typename PartitionType>
  void
  BuildLocalHistograms(size_t page_idx,
                       GHistIndexMatrix const &gidx,
                       const std::vector<GradientPair> &gpair_h,
                       int depth,
                       const common::ColumnMatrix& column_matrix,
                       const PartitionType* p_opt_partition_builder,
                       // template?
                       std::vector<uint16_t>* p_node_ids) {
    const PartitionType& opt_partition_builder = *p_opt_partition_builder;
    std::vector<uint16_t>& node_ids = *p_node_ids;
    int nthreads = this->n_threads_;
    std::vector<int> fids;
    // now column sampling supported only for missings due to fid_least_bins_ set
    if (column_sampling) {
      const size_t n_sampled_features = column_sampler_->GetFeatureSet(depth)->Size();
      fids.resize(n_sampled_features, 0);
      for (size_t i = 0; i < n_sampled_features; ++i) {
        fids[i] = column_sampler_->GetFeatureSet(depth)->ConstHostVector()[i];
      }
    }
    // for(size_t tid = 0; tid < nthreads; ++tid)
    const size_t n_features = gidx.cut.Ptrs().size() - 1;
    const uint32_t* offsets = gidx.index.Offset();
    #pragma omp parallel num_threads(nthreads)
    {
      size_t tid = omp_get_thread_num();
      const std::vector<common::Slice>& local_slices =
        opt_partition_builder.GetSlices(tid);
      buffer_.template AllocateHistForLocalThread<any_missing>(
        opt_partition_builder.node_id_for_threads[tid], tid,
        n_features, offsets);
      for (const common::Slice& slice : local_slices) {
        BuildLocalHistogramsForSlice<BinIdxType, any_missing, hist_fit_to_l2, column_sampling>
                                    (gpair_h, gidx, slice, depth, node_ids, tid, column_matrix, fids);
      }
    }
  }

  void
  AddHistRows(int *starting_index, int *sync_count,
              std::vector<ExpandEntry> const &nodes_for_explicit_hist_build,
              std::vector<ExpandEntry> const &nodes_for_subtraction_trick,
              RegTree *p_tree) {
    if (is_distributed_) {
      this->AddHistRowsDistributed(starting_index, sync_count,
                                   nodes_for_explicit_hist_build,
                                   nodes_for_subtraction_trick, p_tree);
    } else {
      this->AddHistRowsLocal(starting_index, sync_count,
                             nodes_for_explicit_hist_build,
                             nodes_for_subtraction_trick);
    }
  }

  /** Main entry point of this class, build histogram for tree nodes. */
  template <typename BinIdxType, bool any_missing, bool hist_fit_to_l2, typename PartitionType>
  void BuildHist(size_t page_id, GHistIndexMatrix const &gidx,
                 RegTree *p_tree,
                 std::vector<GradientPair> const &gpair,
                 int depth, const common::ColumnMatrix& column_matrix,
                 std::vector<ExpandEntry> const &nodes_for_explicit_hist_build,
                 std::vector<ExpandEntry> const &nodes_for_subtraction_trick,
                 const PartitionType* p_opt_partition_builder,
                 std::vector<uint16_t>* p_node_ids,
                 const std::vector<std::vector<uint16_t>>* merged_thread_ids = nullptr) {
    int starting_index = std::numeric_limits<int>::max();
    int sync_count = 0;
    if (page_id == 0) {
      this->AddHistRows(&starting_index, &sync_count,
                        nodes_for_explicit_hist_build,
                        nodes_for_subtraction_trick, p_tree);
    }

    //TODO(razdoburdin): Verify naming.
    bool column_sampling = !any_missing || 
                            (any_missing && 
                            (colsample_bytree_ < 0.1 || colsample_bylevel_ < 0.1) && 
                            !column_matrix.AnySparseColumn() &&
                            colsample_bynode_ == 1);
    if (column_sampling) {
      BuildLocalHistograms<BinIdxType, any_missing, 
                           hist_fit_to_l2, true>(page_id, gidx, gpair, depth, column_matrix,
                                                 p_opt_partition_builder,
                                                 p_node_ids);
    } else {
      BuildLocalHistograms<uint32_t, any_missing,
                           hist_fit_to_l2, false>(page_id, gidx, gpair, depth, column_matrix,
                                                  p_opt_partition_builder,
                                                  p_node_ids);
    }

    CHECK_GE(n_batches_, 1);
    if (page_id != n_batches_ - 1) {
      return;
    }

    if (is_distributed_) {
      this->SyncHistogramDistributed(p_tree, nodes_for_explicit_hist_build,
                                     nodes_for_subtraction_trick,
                                     starting_index, sync_count,
                                     p_opt_partition_builder, merged_thread_ids);
    } else {
      this->SyncHistogramLocal(p_tree, nodes_for_explicit_hist_build,
                               nodes_for_subtraction_trick, starting_index,
                               sync_count, p_opt_partition_builder, merged_thread_ids);
    }
  }

  template <typename PartitionType>
  void SyncHistogramDistributed(
      RegTree *p_tree,
      std::vector<ExpandEntry> const &nodes_for_explicit_hist_build,
      std::vector<ExpandEntry> const &nodes_for_subtraction_trick,
      int starting_index, int sync_count,
      const PartitionType* p_opt_partition_builder,
      const std::vector<std::vector<uint16_t>>* merged_thread_ids = nullptr) {
    const PartitionType& opt_partition_builder = *p_opt_partition_builder;
    const size_t nbins = hist_.GetNumBins();
    common::BlockedSpace2d space(
        nodes_for_explicit_hist_build.size(), [&](size_t) { return nbins; },
        1024);
    common::ParallelFor2d(
        space, n_threads_, [&](size_t node, common::Range1d r) {
          const auto &entry = nodes_for_explicit_hist_build[node];
          auto this_hist = this->hist_[entry.nid];
          // Merging histograms from each thread into once
          if (merged_thread_ids) {
            this->buffer_.ReduceHist(reinterpret_cast<GradientSumT*>(this_hist.data()),
                                    (*merged_thread_ids)[node],
                                    entry.nid, r.begin(), r.end());
          } else {
            this->buffer_.ReduceHist(reinterpret_cast<GradientSumT*>(this_hist.data()),
                                    opt_partition_builder.GetThreadIdsForNode(entry.nid),
                                    entry.nid, r.begin(), r.end());
          }
          // Store posible parent node
          auto this_local = hist_local_worker_[entry.nid];
          common::CopyHist(this_local, this_hist, r.begin(), r.end());

          if (!(*p_tree)[entry.nid].IsRoot()) {
            const size_t parent_id = (*p_tree)[entry.nid].Parent();
            const int subtraction_node_id =
                nodes_for_subtraction_trick[node].nid;
            auto parent_hist = this->hist_local_worker_[parent_id];
            auto sibling_hist = this->hist_[subtraction_node_id];
            common::SubtractionHist(sibling_hist, parent_hist, this_hist,
                                    r.begin(), r.end());
            // Store posible parent node
            auto sibling_local = hist_local_worker_[subtraction_node_id];
            common::CopyHist(sibling_local, sibling_hist, r.begin(), r.end());
          }
        });

    reducer_.Allreduce(this->hist_[starting_index].data(),
                       hist_.GetNumBins() * sync_count);

    ParallelSubtractionHist(space, nodes_for_explicit_hist_build,
                            nodes_for_subtraction_trick, p_tree);

    common::BlockedSpace2d space2(
        nodes_for_subtraction_trick.size(), [&](size_t) { return nbins; },
        1024);
    ParallelSubtractionHist(space2, nodes_for_subtraction_trick,
                            nodes_for_explicit_hist_build, p_tree);
  }

  template <typename PartitionType>
  void SyncHistogramLocal(
      RegTree *p_tree,
      std::vector<ExpandEntry> const &nodes_for_explicit_hist_build,
      std::vector<ExpandEntry> const &nodes_for_subtraction_trick,
      int starting_index, int sync_count,
      const PartitionType* p_opt_partition_builder,
      const std::vector<std::vector<uint16_t>>* merged_thread_ids = nullptr) {
    const PartitionType& opt_partition_builder = *p_opt_partition_builder;
    const size_t nbins = this->hist_.GetNumBins();
    common::BlockedSpace2d space(
        nodes_for_explicit_hist_build.size(), [&](size_t) { return nbins; },
        1024);

    common::ParallelFor2d(
        space, this->n_threads_, [&](size_t node, common::Range1d r) {
          const auto &entry = nodes_for_explicit_hist_build[node];
          auto this_hist = this->hist_[entry.nid];
          // Merging histograms from each thread into once
          if (merged_thread_ids) {
            this->buffer_.ReduceHist(reinterpret_cast<GradientSumT*>(this_hist.data()),
                                    (*merged_thread_ids)[node],
                                    entry.nid, r.begin(), r.end());
          } else {
            // std::cout << "this->buffer_.ReduceHist:" << std::endl;
            this->buffer_.ReduceHist(reinterpret_cast<GradientSumT*>(this_hist.data()),
                                    opt_partition_builder.GetThreadIdsForNode(entry.nid),
                                    entry.nid, r.begin(), r.end());
            // std::cout << "this->buffer_.ReduceHist: finished" << std::endl;
          }
          if (!(*p_tree)[entry.nid].IsRoot()) {
            const size_t parent_id = (*p_tree)[entry.nid].Parent();
            const int subtraction_node_id =
                nodes_for_subtraction_trick[node].nid;
            auto parent_hist = this->hist_[parent_id];
            auto sibling_hist = this->hist_[subtraction_node_id];
            common::SubtractionHist(sibling_hist, parent_hist, this_hist,
                                    r.begin(), r.end());
          }
        });
  }

 public:
  /* Getters for tests. */
  common::HistCollection<GradientSumT> const& Histogram() {
    return hist_;
  }
  auto& Buffer() { return buffer_; }

 private:
  void
  ParallelSubtractionHist(const common::BlockedSpace2d &space,
                          const std::vector<ExpandEntry> &nodes,
                          const std::vector<ExpandEntry> &subtraction_nodes,
                          const RegTree *p_tree) {
    common::ParallelFor2d(
        space, this->n_threads_, [&](size_t node, common::Range1d r) {
          const auto &entry = nodes[node];
          if (!((*p_tree)[entry.nid].IsLeftChild())) {
            auto this_hist = this->hist_[entry.nid];

            if (!(*p_tree)[entry.nid].IsRoot()) {
              const int subtraction_node_id = subtraction_nodes[node].nid;
              auto parent_hist = hist_[(*p_tree)[entry.nid].Parent()];
              auto sibling_hist = hist_[subtraction_node_id];
              common::SubtractionHist(this_hist, parent_hist, sibling_hist,
                                      r.begin(), r.end());
            }
          }
        });
  }

  // Add a tree node to histogram buffer in local training environment.
  void AddHistRowsLocal(
      int *starting_index, int *sync_count,
      std::vector<ExpandEntry> const &nodes_for_explicit_hist_build,
      std::vector<ExpandEntry> const &nodes_for_subtraction_trick) {
    for (auto const &entry : nodes_for_explicit_hist_build) {
      int nid = entry.nid;
      this->hist_.AddHistRow(nid);
      (*starting_index) = std::min(nid, (*starting_index));
    }
    (*sync_count) = nodes_for_explicit_hist_build.size();

    for (auto const &node : nodes_for_subtraction_trick) {
      this->hist_.AddHistRow(node.nid);
    }
    this->hist_.AllocateAllData();
  }

  void AddHistRowsDistributed(
      int *starting_index, int *sync_count,
      std::vector<ExpandEntry> const &nodes_for_explicit_hist_build,
      std::vector<ExpandEntry> const &nodes_for_subtraction_trick,
      RegTree *p_tree) {
    const size_t explicit_size = nodes_for_explicit_hist_build.size();
    const size_t subtaction_size = nodes_for_subtraction_trick.size();
    std::vector<int> merged_node_ids(explicit_size + subtaction_size);
    for (size_t i = 0; i < explicit_size; ++i) {
      merged_node_ids[i] = nodes_for_explicit_hist_build[i].nid;
    }
    for (size_t i = 0; i < subtaction_size; ++i) {
      merged_node_ids[explicit_size + i] = nodes_for_subtraction_trick[i].nid;
    }
    std::sort(merged_node_ids.begin(), merged_node_ids.end());
    int n_left = 0;
    for (auto const &nid : merged_node_ids) {
      if ((*p_tree)[nid].IsLeftChild()) {
        this->hist_.AddHistRow(nid);
        (*starting_index) = std::min(nid, (*starting_index));
        n_left++;
        this->hist_local_worker_.AddHistRow(nid);
      }
    }
    for (auto const &nid : merged_node_ids) {
      if (!((*p_tree)[nid].IsLeftChild())) {
        this->hist_.AddHistRow(nid);
        this->hist_local_worker_.AddHistRow(nid);
      }
    }
    this->hist_.AllocateAllData();
    this->hist_local_worker_.AllocateAllData();
    (*sync_count) = std::max(1, n_left);
  }
};
}      // namespace tree
}      // namespace xgboost
#endif  // XGBOOST_TREE_HIST_HISTOGRAM_H_
