/*!
 * Copyright 2017-2024 by Contributors
 * \file hist_synchronizer.h
 */
#ifndef PLUGIN_SYCL_TREE_HIST_SYNCHRONIZER_H_
#define PLUGIN_SYCL_TREE_HIST_SYNCHRONIZER_H_

#include <vector>

#include "../common/hist_util.h"
#include "expand_entry.h"

namespace xgboost {
namespace sycl {
namespace tree {

template <typename GradientSumT>
class HistUpdater;

template <typename GradientSumT>
class HistSynchronizer {
 public:
  virtual void SyncHistograms(HistUpdater<GradientSumT>* builder,
                              const std::vector<int>& sync_ids,
                              const xgboost::RegTree& tree,
                              const std::vector<::sycl::event>& events_subtraction,
                              const std::vector<::sycl::event>& events_build,
                              std::vector<::sycl::event>* events_out) = 0;
  virtual ~HistSynchronizer() = default;
};

template <typename GradientSumT>
class BatchHistSynchronizer: public HistSynchronizer<GradientSumT> {
 public:
  void SyncHistograms(HistUpdater<GradientSumT>* builder,
                      const std::vector<int>& sync_ids,
                      const xgboost::RegTree& tree,
                      const std::vector<::sycl::event>& events_subtraction,
                      const std::vector<::sycl::event>& events_build,
                      std::vector<::sycl::event>* events_out) override {
    const size_t nbins = builder->hist_builder_.GetNumBins();

    for (int i = 0; i < builder->nodes_for_explicit_hist_build_.size(); i++) {
      const auto entry = builder->nodes_for_explicit_hist_build_[i];
      auto& this_hist = builder->hist_[entry.nid];

      if (!tree[entry.nid].IsRoot()) {
        const size_t parent_id = tree[entry.nid].Parent();
        auto& parent_hist = builder->hist_[parent_id];
        auto& sibling_hist = builder->hist_[entry.GetSiblingId(tree, parent_id)];
        events_out->push_back(common::SubtractionHist(builder->qu_, &sibling_hist, parent_hist,
                                                      this_hist, nbins,
                                                      {events_subtraction[i], events_build[i]}));
      } else {
        events_out->push_back(events_build[i]);
      }
    }
  }
};

template <typename GradientSumT>
class DistributedHistSynchronizer: public HistSynchronizer<GradientSumT> {
 public:
  void SyncHistograms(HistUpdater<GradientSumT>* builder,
                      const std::vector<int>& sync_ids,
                      const xgboost::RegTree& tree,
                      const std::vector<::sycl::event>& events_subtraction,
                      const std::vector<::sycl::event>& events_build,
                      std::vector<::sycl::event>* events_out) override {
    const size_t nbins = builder->hist_builder_.GetNumBins();
    for (int node = 0; node < builder->nodes_for_explicit_hist_build_.size(); node++) {
      const auto entry = builder->nodes_for_explicit_hist_build_[node];
      auto& this_hist = builder->hist_[entry.nid];
      // // Store posible parent node
      auto& this_local = builder->hist_local_worker_[entry.nid];
      common::CopyHist(builder->qu_, &this_local, this_hist, nbins);

      if (!tree[entry.nid].IsRoot()) {
        const size_t parent_id = tree[entry.nid].Parent();
        auto sibling_nid = entry.GetSiblingId(tree, parent_id);
        auto& parent_hist = builder->hist_local_worker_[parent_id];

        auto& sibling_hist = builder->hist_[sibling_nid];
        common::SubtractionHist(builder->qu_, &sibling_hist, parent_hist,
                                this_hist, nbins, {events_subtraction[node], events_build[node]});
        builder->qu_->wait_and_throw();
        // Store posible parent node
        auto& sibling_local = builder->hist_local_worker_[sibling_nid];
        common::CopyHist(builder->qu_, &sibling_local, sibling_hist, nbins);
      }
    }
    builder->ReduceHists(sync_ids, nbins);

    ParallelSubtractionHist(builder, builder->nodes_for_explicit_hist_build_, tree,
                            events_subtraction, events_build);
    ParallelSubtractionHist(builder, builder->nodes_for_subtraction_trick_, tree,
                            events_subtraction, events_build);
  }

  void ParallelSubtractionHist(HistUpdater<GradientSumT>* builder,
                               const std::vector<ExpandEntry>& nodes,
                               const xgboost::RegTree& tree,
                               const std::vector<::sycl::event>& events_subtraction,
                               const std::vector<::sycl::event>& events_build) {
    const size_t nbins = builder->hist_builder_.GetNumBins();
    for (int node = 0; node < nodes.size(); node++) {
      const auto entry = nodes[node];
      if (!(tree[entry.nid].IsLeftChild())) {
        auto& this_hist = builder->hist_[entry.nid];

        if (!tree[entry.nid].IsRoot()) {
          const size_t parent_id = tree[entry.nid].Parent();
          auto& parent_hist = builder->hist_[parent_id];
          auto& sibling_hist = builder->hist_[entry.GetSiblingId(tree, parent_id)];
          common::SubtractionHist(builder->qu_, &this_hist, parent_hist,
                                  sibling_hist, nbins, {events_subtraction[node], events_build[node]});
          builder->qu_->wait_and_throw();
        }
      }
    }
  }

 private:
  std::vector<::sycl::event> hist_sync_events_;
};

}  // namespace tree
}  // namespace sycl
}  // namespace xgboost

#endif  // PLUGIN_SYCL_TREE_HIST_SYNCHRONIZER_H_
