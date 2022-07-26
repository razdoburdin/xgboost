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


enum class ContainerType : std::uint8_t {  // NOLINT
  kVector = 0,
  kUnorderedMap = 1
};
// Standartize interface for acces vector and unorderd_map
template <typename T>
class FlexibleContainer {
 public:
  void ResizeIfSmaller(size_t size) {
    if (type_ == ContainerType::kVector) {
      vector_.resize(vector_.size() < size ? size : vector_.size());
    }
  }

  template <ContainerType container_type>
  void Increment(size_t idx, T val) {
    if (container_type == ContainerType::kVector) {
      vector_[idx] += val;
    } else {
      unordered_map_[idx] += val;
    }
  }

  T& operator[](size_t idx) {
    if (type_ == ContainerType::kVector) {
      return vector_[idx];
    } else {
      return unordered_map_[idx];
    }
  }

  T& At(size_t idx) const {
    if (type_ == ContainerType::kVector) {
      return vector_.at(idx);
    } else {
      return unordered_map_.at(idx);
    }
  }

  std::vector<uint32_t> GetUniqueIdx() {
    std::vector<uint32_t> unique_idx;
    if (type_ == ContainerType::kVector) {
      for (size_t num = 0; num < vector_.size(); ++num) {
        if (vector_[num] > 0) {
          unique_idx.push_back(num);
        }
      }
    } else {
      unique_idx.resize(unordered_map_.size(), 0);
      size_t i = 0;
      for (const auto& tnc : unordered_map_) {
        unique_idx[i++] = tnc.first;
      }
    }
    return unique_idx;
  }

  void Clear() {
    if (type_ == ContainerType::kVector) {
      vector_.clear();
    } else {
      unordered_map_.clear();
    }
  }

  ContainerType GetContainerType() const {
    return type_;
  }

  void SetContainerType(ContainerType type) {
    type_ = type;
  }

 private:
  std::unordered_map<uint32_t, T> unordered_map_;
  std::vector<T> vector_;
  ContainerType type_ = ContainerType::kVector;
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
    }

    std::vector<Slice> addr;
    std::vector<uint16_t> nodes_id;
    std::vector<uint32_t> rows_nodes_wise;

    FlexibleContainer<uint32_t> nodes_count;
    FlexibleContainer<NodesCountRange> nodes_count_range;

    std::vector<uint32_t> vec_rows;
    std::vector<uint32_t> vec_rows_remain;

    std::unordered_map<uint32_t, size_t> states;
    std::unordered_map<uint32_t, bool> default_flags;
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
    if (nodes_[nid].threads_id.empty()) {
      nodes_[nid].threads_id.push_back(tid);
      threads_[tid].nodes_id.push_back(nid);
    } else {
      if (nodes_[nid].threads_id.back() != tid) {
        nodes_[nid].threads_id.push_back(tid);
        threads_[tid].nodes_id.push_back(nid);
      }
    }
  }

  void Init(size_t n_threads) {
    threads_.resize(n_threads);
  }

  void Init(size_t n_threads, size_t chunck_size, bool is_loss_guided) {
    Init(n_threads);
    nodes_.clear();

    if (GetThreadInfoPtr(0)->vec_rows.size() == 0) {
    #pragma omp parallel num_threads(n_threads)
      {
        size_t tid = omp_get_thread_num();
        auto thread_info = GetThreadInfoPtr(tid);
        if (thread_info->vec_rows.size() == 0) {
          thread_info->vec_rows.resize(chunck_size + 2, 0);
          if (is_loss_guided) {
            thread_info->vec_rows_remain.resize(chunck_size + 2, 0);
          }
        }
      }
    }
    // ForEachThread([](auto& ti) {ti.nodes_count_range.Clear();});
  }

  bool HasNodeInfo(uint32_t nid) const {
    return nodes_.find(nid) != nodes_.end();
  }

  const NodeInfo* GetNodeInfoPtr(uint32_t nid) const {
    return &(nodes_.at(nid));
  }

  NodeInfo* GetNodeInfoPtr(uint32_t nid) {
    return &(nodes_[nid]);
  }

  const ThreadInfo* GetThreadInfoPtr(size_t tid) const {
    return &(threads_.at(tid));
  }

  ThreadInfo* GetThreadInfoPtr(size_t tid) {
    return const_cast<ThreadInfo*>(const_cast<const ThreadsManager*>(this)->GetThreadInfoPtr(tid));
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
  std::unordered_map<uint32_t, NodeInfo> nodes_;

  CyclicView<ThreadInfo> threads_cyclic_view_;
  SaturationView<ThreadInfo> threads_saturation_view_;
};

}  // namespace common
}  // namespace xgboost

#endif  // XGBOOST_COMMON_THREADS_MANAGER_H_
