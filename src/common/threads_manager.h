/*!
 * Copyright 2022 by Contributors
 * \file threads_manager.h
 * \brief Helper class for control threads in partition builder
 */
#ifndef XGBOOST_COMMON_THREADS_MANAGER_H_
#define XGBOOST_COMMON_THREADS_MANAGER_H_

#include <utility>
#include <vector>
#include <unordered_map>

#include "flexible_container.h"

namespace xgboost {
namespace common {

struct Slice {
  uint32_t* addr {nullptr};
  uint32_t b {0};
  uint32_t e {0};
  uint32_t Size() const {
    CHECK_GE(e, b);
    return e - b;
  }
};

// CRTP
template <class DataT,
          class DerivedView>
class AbstractView {
 protected:
  std::vector<DataT>* storage_ptr_;
  size_t counter_ = 0;

  explicit AbstractView(std::vector<DataT>* storage_ptr) : storage_ptr_(storage_ptr) {}

 public:
  size_t Idx() const {
    return static_cast<const DerivedView*>(this)->Idx();
  }

  void Reset() {
    counter_ = 0;
  }

  DataT* GetItem() const {
    return &(storage_ptr_->at(Idx()));
  }

  DataT* NextItem() {
    ++counter_;
    return GetItem();
  }
};

/*
 * This class provides a cyclic access to the data placed in external linear container of type std::vector<DataT>.
 * <!> Be carefull in case of external container is copying or deleting.
 */
template <class DataT>
class CyclicView : public AbstractView<DataT, CyclicView<DataT>> {
 public:
  explicit CyclicView(std::vector<DataT>* storage_ptr) :
    AbstractView<DataT, CyclicView<DataT>>(storage_ptr) {}

  size_t Idx() const {
    return this->counter_ % this->storage_ptr_->size();
  }

  size_t CycleIdx() const {
    return this->counter_ / this->storage_ptr_->size();
  }
};

/*
 * This class provides an access with saturation to the data placed in external linear container of type std::vector<DataT>.
 * Access with saturation means that in case of counter is higher than size - 1, than the last element is returning.
 * <!> Be carefull in case of external container is copying or deleting.
 */
template <class DataT>
class SaturationView : public AbstractView<DataT, SaturationView<DataT>> {
 public:
  explicit SaturationView(std::vector<DataT>* storage_ptr) :
    AbstractView<DataT, SaturationView<DataT>>(storage_ptr) {}

  size_t Idx() const {
    return this->counter_ < this->storage_ptr_->size() - 1
           ? this->counter_
           : this->storage_ptr_->size() - 1;
  }
};

class ThreadsManager {
 public:
  struct ThreadInfo {
    struct NodesCountRange {
      uint32_t begin;
      uint32_t end;
    };

    NodesCountRange* GetNodesCountRangePtr(size_t nid) {
      return &(nodes_count_range[nid]);
    }

    void SetContainersType(ContainerType type) {
      nodes_count.SetContainerType(type);
      nodes_count_range.SetContainerType(type);
      states.SetContainerType(type);
      default_flags.SetContainerType(type);
      counts.SetContainerType(type);
    }

    std::vector<Slice> addr;
    std::vector<uint16_t> nodes_id;
    std::vector<uint32_t> rows_nodes_wise;

    FlexibleContainer<uint32_t> nodes_count;
    FlexibleContainer<NodesCountRange> nodes_count_range;

    std::vector<uint32_t> vec_rows;
    std::vector<uint32_t> vec_rows_remain;

    FlexibleContainer<size_t> states;
    FlexibleContainer<uint8_t> default_flags;

    FlexibleContainer<uint32_t> counts;
  };

  struct NodeInfo {
    std::vector<uint16_t> threads_id;
  };

  ThreadsManager() : threads_(), nodes_(), threads_cyclic_view_(&threads_),
                                           threads_saturation_view_(&threads_) {}

  ThreadsManager(const ThreadsManager& other) : threads_(other.threads_), nodes_(other.nodes_),
                                                threads_cyclic_view_(&threads_),
                                                threads_saturation_view_(&threads_) {}

  template <class Predicate>
  void ForEachThread(Predicate&& predicate) {
      std::for_each(threads_.begin(), threads_.end(), predicate);
  }

  template <class Predicate>
  void ForEachNode(Predicate&& predicate) {
      std::for_each(nodes_.begin(), nodes_.end(), predicate);
  }

  /* Tie thread_id with node_id */
  void Tie(size_t tid, uint32_t nid) {
    CHECK_LT(tid, threads_.size());
    auto node_info = GetNodeInfoPtr(nid);
    auto thread_info = GetThreadInfoPtr(tid);
    if (node_info->threads_id.empty()) {
      node_info->threads_id.push_back(tid);
      thread_info->nodes_id.push_back(nid);
    } else {
      if (node_info->threads_id.back() != tid) {
        node_info->threads_id.push_back(tid);
        thread_info->nodes_id.push_back(nid);
      }
    }
  }

  void Init(size_t n_threads, size_t chunck_size, bool is_loss_guided,
            bool use_linear_container, size_t nodes_amount) {
    ContainerType container_type = use_linear_container ?
                                   ContainerType::kVector :
                                   ContainerType::kUnorderedMap;
    nodes_.SetContainerType(container_type);
    nodes_.Clear();
    nodes_.ResizeIfSmaller(nodes_amount);

    threads_.resize(n_threads);
    for (auto& ti : threads_) {
      ti.SetContainersType(container_type);
      ti.vec_rows.resize(chunck_size + 2, 0);
      if (is_loss_guided) {
        ti.vec_rows_remain.resize(chunck_size + 2, 0);
      }
    }
  }

  const NodeInfo* GetNodeInfoPtr(uint32_t nid) const {
    return &(nodes_[nid]);
  }

  NodeInfo* GetNodeInfoPtr(uint32_t nid) {
    return &(nodes_[nid]);
  }

  const ThreadInfo* GetThreadInfoPtr(size_t tid) const {
    return &(threads_.at(tid));
  }

  ThreadInfo* GetThreadInfoPtr(size_t tid) {
    return &(threads_[tid]);
  }

  CyclicView<ThreadInfo> GetThreadsCyclicView() {
    threads_cyclic_view_.Reset();
    return threads_cyclic_view_;
  }

  SaturationView<ThreadInfo> GetThreadsSaturationView() {
    threads_saturation_view_.Reset();
    return threads_saturation_view_;
  }

  size_t NumThreads() const {
    return threads_.size();
  }

  size_t AccumulateVecRows(size_t end) {
    size_t ans = 0;
    for (size_t i = 0; i < end; ++i) {
      auto thread_info = GetThreadInfoPtr(i);
      ans += thread_info->vec_rows[0];
    }
    return ans;
  }

  size_t AccumulateVecRowsRemain(size_t end) {
    size_t ans = 0;
    for (size_t i = 0; i < end; ++i) {
      auto thread_info = GetThreadInfoPtr(i);
      ans += thread_info->vec_rows_remain[0];
    }
    return ans;
  }

 private:
  std::vector<ThreadInfo> threads_;
  FlexibleContainer<NodeInfo> nodes_;

  CyclicView<ThreadInfo> threads_cyclic_view_;
  SaturationView<ThreadInfo> threads_saturation_view_;
};

}  // namespace common
}  // namespace xgboost

#endif  // XGBOOST_COMMON_THREADS_MANAGER_H_
