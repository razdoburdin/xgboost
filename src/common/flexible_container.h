/*!
 * Copyright 2022 by Contributors
 * \file flexible_container.h
 * \brief Helper class for control threads in partition builder
 */
#ifndef XGBOOST_COMMON_FLEXIBLE_CONTAINER_H_
#define XGBOOST_COMMON_FLEXIBLE_CONTAINER_H_

#include <utility>
#include <vector>
#include <unordered_map>

namespace xgboost {
namespace common {

enum class ContainerType : std::uint8_t {  // NOLINT
  kVector = 0,
  kUnorderedMap = 1
};
// Standartize interface for acces vector and unorderd_map
template <typename T>
class FlexibleContainer {
 public:
  class Iterator {
    typename std::vector<T>::iterator vector_it_;
    typename std::unordered_map<uint32_t, T>::iterator unordered_map_it_;
    ContainerType type_ = ContainerType::kVector;

   public:
    explicit Iterator(typename std::vector<T>::iterator it) {
      vector_it_ = it;
      type_ = ContainerType::kVector;
    }

    explicit Iterator(typename std::unordered_map<uint32_t, T>::iterator it) {
      unordered_map_it_ = it;
      type_ = ContainerType::kUnorderedMap;
    }

    T& operator*() { return type_ == ContainerType::kVector ?
                                     *vector_it_ :
                                     unordered_map_it_->second;}

    bool operator != (const Iterator& rhs) {
      if (type_ != rhs.type_) {
        return true;
      }
      if (type_ == ContainerType::kVector) {
        return vector_it_ != rhs.vector_it_;
      } else {
        return unordered_map_it_ != rhs.unordered_map_it_;
      }
    }
    void operator ++() {
      if (type_ == ContainerType::kVector) {
          ++vector_it_;
      } else {
        ++unordered_map_it_;
      }
    }
  };

  auto begin() {
    if (type_ == ContainerType::kVector) {
      return Iterator(vector_.begin());
    } else {
      return Iterator(unordered_map_.begin());
    }
  }

  auto end() {
    if (type_ == ContainerType::kVector) {
      return Iterator(vector_.end());
    } else {
      return Iterator(unordered_map_.end());
    }
  }

  void ResizeIfSmaller(size_t size) {
    if (type_ == ContainerType::kVector) {
      vector_.resize(vector_.size() < size ? size : vector_.size());
    }
  }

  void Increment(size_t idx, T val) {
    if (type_ == ContainerType::kVector) {
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

  const T& operator[](size_t idx) const {
    if (type_ == ContainerType::kVector) {
      return vector_[idx];
    } else {
      if (unordered_map_.count(idx) > 0) {
        return unordered_map_.at(idx);
      } else {
        static T empty_item;
        return empty_item;
      }
    }
  }

  const T& At(size_t idx) const {
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

}  // namespace common
}  // namespace xgboost

#endif  // XGBOOST_COMMON_FLEXIBLE_CONTAINER_H_