/*!
 * Copyright 2017-2024 by Contributors
 * \file hist_row_adder.h
 */
#ifndef PLUGIN_SYCL_TREE_HIST_ROW_ADDER_H_
#define PLUGIN_SYCL_TREE_HIST_ROW_ADDER_H_

#include <vector>
#include <algorithm>

namespace xgboost {
namespace sycl {
namespace tree {

template <typename GradientSumT>
class HistRowsAdder {
 public:
  virtual void AddHistRows(HistUpdater<GradientSumT>* builder,
                           std::vector<int>* sync_ids, const xgboost::RegTree& tree,
                           std::vector<::sycl::event>* events_explicit,
                           std::vector<::sycl::event>* events_subtraction) = 0;
  virtual ~HistRowsAdder() = default;
};

template <typename GradientSumT>
class BatchHistRowsAdder: public HistRowsAdder<GradientSumT> {
 public:
  void AddHistRows(HistUpdater<GradientSumT>* builder,
                   std::vector<int>* sync_ids,const xgboost::RegTree& tree,
                   std::vector<::sycl::event>* events_explicit,
                   std::vector<::sycl::event>* events_subtraction) override {
    for (auto const& entry : builder->nodes_for_explicit_hist_build_) {
      int nid = entry.nid;
      auto event = builder->hist_.AddHistRow(nid);
      events_explicit->push_back(event);
    }
    for (auto const& node : builder->nodes_for_subtraction_trick_) {
      auto event = builder->hist_.AddHistRow(node.nid);
      events_subtraction->push_back(event);
    }
  }
};


template <typename GradientSumT>
class DistributedHistRowsAdder: public HistRowsAdder<GradientSumT> {
 public:
  void AddHistRows(HistUpdater<GradientSumT>* builder,
                   std::vector<int>* sync_ids, const xgboost::RegTree& tree,
                   std::vector<::sycl::event>* events_explicit,
                   std::vector<::sycl::event>* events_subtraction) override {
    const size_t explicit_size = builder->nodes_for_explicit_hist_build_.size();
    const size_t subtaction_size = builder->nodes_for_subtraction_trick_.size();
    std::vector<int> merged_node_ids(explicit_size + subtaction_size);
    for (size_t i = 0; i < explicit_size; ++i) {
      merged_node_ids[i] = builder->nodes_for_explicit_hist_build_[i].nid;
    }
    for (size_t i = 0; i < subtaction_size; ++i) {
      merged_node_ids[explicit_size + i] =
      builder->nodes_for_subtraction_trick_[i].nid;
    }
    std::sort(merged_node_ids.begin(), merged_node_ids.end());
    sync_ids->clear();
    for (auto const& nid : merged_node_ids) {
      if (tree[nid].IsLeftChild()) {
        auto event0 = builder->hist_.AddHistRow(nid);
        events_explicit->push_back(event0);

        auto event1 = builder->hist_local_worker_.AddHistRow(nid);
        events_explicit->push_back(event1);
        sync_ids->push_back(nid);
      }
    }
    for (auto const& nid : merged_node_ids) {
      if (!(tree[nid].IsLeftChild())) {
        auto event0 = builder->hist_.AddHistRow(nid);
        events_subtraction->push_back(event0);

        auto event1 = builder->hist_local_worker_.AddHistRow(nid);
        events_subtraction->push_back(event1);
      }
    }
  }
};

}  // namespace tree
}  // namespace sycl
}  // namespace xgboost

#endif  // PLUGIN_SYCL_TREE_HIST_ROW_ADDER_H_
