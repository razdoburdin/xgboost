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
    return e - b;
  }
};

template <class DataT>
class AbstractBuffer {
 protected:
  std::vector<DataT>* storage_ptr_;
  size_t counter_ = 0;

 public:
  explicit AbstractBuffer(std::vector<DataT>* storage_ptr) : storage_ptr_(storage_ptr) {}

  AbstractBuffer(const AbstractBuffer&) = delete;
  AbstractBuffer& operator=(const AbstractBuffer&) = delete;
  virtual ~AbstractBuffer() = default;

  virtual size_t Id() const = 0;

  void Reset() {
    counter_ = 0;
  }

  DataT* GetItem() const {
    return &(storage_ptr_->at(Id()));
  }

  DataT* NextItem() {
    ++counter_;
    return GetItem();
  }
};

/*
 * This class provide a cyclic access to the data placed in external linear container of type std::vector<DataT>.
 * <!> Be carefull in case of external container is copying or deleting.
 */
template <class DataT>
class CyclicBuffer : public AbstractBuffer<DataT> {
 public:
  explicit CyclicBuffer(std::vector<DataT>* storage_ptr) : AbstractBuffer<DataT>(storage_ptr) {}

  size_t Id() const override {
    return this->counter_ % this->storage_ptr_->size();
  }

  size_t NCycles() const {
    return this->counter_ / this->storage_ptr_->size();
  }
};

/*
 * This class provide an access with saturation to the data placed in external linear container of type std::vector<DataT>.
 * Access with saturation means that in case of counter is higher than size - 1, than the last element is returning.
 * <!> Be carefull in case of external container is copying or deleting.
 */
template <class DataT>
class SaturationBuffer : public AbstractBuffer<DataT> {
 public:
  explicit SaturationBuffer(std::vector<DataT>* storage_ptr) : AbstractBuffer<DataT>(storage_ptr) {}

  size_t Id() const override {
    return this->counter_ < this->storage_ptr_->size() - 1
           ? this->counter_
           : this->storage_ptr_->size() - 1;
  }
};

class ThreadsManager {
 public:
  struct ThreadInfo {
    std::vector<Slice> addr;
    std::vector<uint16_t> nodes_id;
    std::unordered_map<uint32_t, uint32_t> nodes_count;
    std::vector<uint32_t> rows_nodes_wise;

    struct NodesCountRange {
      uint32_t begin;
      uint32_t end;
    };

    std::unordered_map<uint32_t, NodesCountRange> nodes_count_range;
    NodesCountRange* GetNodesCountRangePtr(size_t nid) {
     return &(nodes_count_range[nid]);
    }

    std::vector<uint32_t> vec_rows;
    std::vector<uint32_t> vec_rows_remain;

    std::unordered_map<uint32_t, size_t> states;
    std::unordered_map<uint32_t, bool> default_flags;
  };

  struct NodeInfo {
    std::vector<uint16_t> threads_id;
  };

  ThreadsManager() : threads(), nodes(), threads_cbuffer(&threads), threads_sbuffer(&threads) {}

  ThreadsManager(const ThreadsManager& other) : threads(other.threads), nodes(other.nodes),
                                                threads_cbuffer(&threads),
                                                threads_sbuffer(&threads) {}

  /* Tie thread_id with node_id */
  void Tie(size_t tid, uint32_t nid) {
    CHECK_LT(tid, threads.size());
    if (nodes[nid].threads_id.empty()) {
      nodes[nid].threads_id.push_back(tid);
      threads[tid].nodes_id.push_back(nid);
    } else {
      if (nodes[nid].threads_id.back() != tid) {
        nodes[nid].threads_id.push_back(tid);
        threads[tid].nodes_id.push_back(nid);
      }
    }
  }

  void Init(size_t n_threads) {
    threads.resize(n_threads);
    nodes.clear();
  }

  ThreadInfo* GetThreadInfoPtr(size_t tid) {
    return &(threads[tid]);
  }

  std::vector<ThreadInfo> threads;
  std::unordered_map<uint32_t, NodeInfo> nodes;

  CyclicBuffer<ThreadInfo> threads_cbuffer;
  SaturationBuffer<ThreadInfo> threads_sbuffer;
};

}  // namespace common
}  // namespace xgboost

#endif  // XGBOOST_COMMON_THREADS_MANAGER_H_
