  /*!
 * Copyright 2022 by Contributors
 * \file opt_partition_builder.cc
 * \brief Quick Utility to compute subset of rows
 */

#include<vector>

#include "../common/column_matrix.h"
#include "../common/opt_partition_builder.h"

namespace xgboost {
namespace common {

template <>
void OptPartitionBuilder::UpdateRowBuffer<true>(
                  const std::vector<uint16_t>& compleate_trees_depth_wise,
                  GHistIndexMatrix const& gmat, size_t n_features, size_t depth,
                  const std::vector<uint16_t>& node_ids_) {
  PrepareToUpdateRowBuffer();
  const int cleft = compleate_trees_depth_wise[0];
  const int cright = compleate_trees_depth_wise[1];
  const int parent_id = (*p_tree)[cleft].Parent();
  CHECK_LT(parent_id, partitions.size());
  const size_t parent_begin = partitions[parent_id].b;
  const size_t parent_size = partitions[parent_id].Size();
  for (uint32_t i = 0; i < n_threads; ++i) {
    summ_size_remain += tm.threads[i].vec_rows_remain[0];
  }
  CHECK_EQ(summ_size + summ_size_remain, parent_size);
  SetSlice(cleft, partitions[parent_id].b, summ_size);
  SetSlice(cright, partitions[parent_id].b + summ_size, summ_size_remain);

  #pragma omp parallel num_threads(n_threads)
  {
    uint32_t tid = omp_get_thread_num();
    auto thread_info = tm.GetThreadInfoPtr(tid);
    uint32_t thread_displace = parent_begin;
    for (size_t id = 0; id < tid; ++id) {
      thread_displace += tm.threads[id].vec_rows[0];
    }
    CHECK_LE(thread_displace + thread_info->vec_rows[0], parent_begin + summ_size);
    std::copy(thread_info->vec_rows.data() + 1,
              thread_info->vec_rows.data() + 1 + thread_info->vec_rows[0],
              row_indices_ptr + thread_displace);
    uint32_t thread_displace_left = parent_begin + summ_size;
    for (size_t id = 0; id < tid; ++id) {
      thread_displace_left += tm.threads[id].vec_rows_remain[0];
    }
    CHECK_LE(thread_displace_left + thread_info->vec_rows_remain[0],
              parent_begin + parent_size);
    std::copy(thread_info->vec_rows_remain.data() + 1,
              thread_info->vec_rows_remain.data() + 1 + thread_info->vec_rows_remain[0],
              row_indices_ptr + thread_displace_left);
  }
}

template <>
void OptPartitionBuilder::UpdateRowBuffer<false>(
                  const std::vector<uint16_t>& compleate_trees_depth_wise,
                  GHistIndexMatrix const& gmat, size_t n_features, size_t depth,
                  const std::vector<uint16_t>& node_ids_) {
  PrepareToUpdateRowBuffer();
  if (NeedsBufferUpdate(gmat, n_features, depth)) {
    #pragma omp parallel num_threads(n_threads)
    {
      size_t tid = omp_get_thread_num();
      auto thread_info = tm.GetThreadInfoPtr(tid);
      if (thread_info->rows_nodes_wise.size() == 0) {
        thread_info->rows_nodes_wise.resize(thread_info->vec_rows.size(), 0);
      }
      std::unordered_map<uint32_t, uint32_t> nc;

      std::vector<uint32_t> unique_node_ids(thread_info->nodes_count.size(), 0);
      size_t i = 0;
      for (const auto& tnc : thread_info->nodes_count) {
        CHECK_LT(i, unique_node_ids.size());
        unique_node_ids[i++] = tnc.first;
      }
      std::sort(unique_node_ids.begin(), unique_node_ids.end());

      size_t cummulative_summ = 0;
      std::unordered_map<uint32_t, uint32_t> counts;
      for (const auto& uni : unique_node_ids) {
        auto nodes_count_range = &(thread_info->nodes_count_range[uni]);
        nodes_count_range->begin = cummulative_summ;
        counts[uni] = cummulative_summ;
        nodes_count_range->end = nodes_count_range->begin + thread_info->nodes_count[uni];
        cummulative_summ += thread_info->nodes_count[uni];
      }

      for (size_t i = 0; i < thread_info->vec_rows[0]; ++i) {
        const uint32_t row_id = thread_info->vec_rows[i + 1];
        const uint16_t check_node_id = node_ids_[row_id];
        const uint32_t nod_id = check_node_id;
        thread_info->rows_nodes_wise[counts[nod_id]++] = row_id;
      }
    }
  }
}

template <>
void OptPartitionBuilder::UpdateThreadsWork<true>(
                  const std::vector<uint16_t>& compleate_trees_depth_wise,
                  GHistIndexMatrix const& gmat, size_t n_features, size_t depth,
                  bool is_left_small, bool check_is_left_small) {
  std::for_each(tm.threads.begin(), tm.threads.end(), [](auto& ti) {ti.addr.clear();});
  const int cleft = compleate_trees_depth_wise[0];
  const int cright = compleate_trees_depth_wise[1];
  uint32_t min_node_size = std::min(summ_size, summ_size_remain);
  uint32_t min_node_id = summ_size <= summ_size_remain ? cleft : cright;
  if (check_is_left_small) {
    min_node_id = is_left_small ? cleft : cright;
    min_node_size = is_left_small ? summ_size : summ_size_remain;
  }
  uint32_t thread_size = std::max(common::GetBlockSize(min_node_size, n_threads),
                          std::min(min_node_size, static_cast<uint32_t>(512)));
  for (size_t tid = 0; tid <  n_threads; ++tid) {
    uint32_t th_begin = thread_size * tid;
    uint32_t th_end = std::min(th_begin + thread_size, min_node_size);
    if (th_end > th_begin) {
      CHECK_LT(min_node_id, partitions.size());
      tm.threads[tid].addr.push_back({row_indices_ptr, partitions[min_node_id].b + th_begin,
                                    partitions[min_node_id].b + th_end});
      tm.nodes[min_node_id].threads_id.push_back(tid);
      tm.threads[tid].nodes_id.push_back(min_node_id);
    }
  }
  std::for_each(tm.threads.begin(), tm.threads.end(), [](auto& ti) {ti.nodes_count.clear();});
  std::for_each(tm.threads.begin(), tm.threads.end(),
              [](auto& ti) {ti.nodes_count_range.clear();});
}

template <>
void OptPartitionBuilder::UpdateThreadsWork<false, true>(
                  const std::vector<uint16_t>& compleate_trees_depth_wise) {
  uint32_t block_size = std::max(common::GetBlockSize(summ_size, n_threads),
                                  std::min(summ_size, static_cast<uint32_t>(512)));
  uint32_t curr_thread_size = block_size;
  uint32_t curr_node_disp = 0;
  tm.threads_cbuffer.Reset();
  auto thread_info = tm.threads_cbuffer.GetItem();
  for (uint32_t i = 0; i < n_threads; ++i) {
    while (curr_thread_size != 0) {
      uint32_t node_id = tm.threads_cbuffer.NCycles();
      const uint32_t curr_thread_node_size = thread_info->nodes_count[node_id];
      auto nodes_count_range = thread_info->GetNodesCountRangePtr(node_id);

      if (curr_thread_node_size == 0) {
        thread_info = tm.threads_cbuffer.NextItem();
      } else {
        uint32_t* slice_addr = thread_info->rows_nodes_wise.data();
        uint32_t slice_begin = nodes_count_range->begin;
        uint32_t slice_end = slice_begin;
        CHECK_EQ(nodes_count_range->begin + curr_thread_node_size,
                  nodes_count_range->end);
        tm.Tie(i, node_id);
        if (curr_thread_node_size > 0 && curr_thread_node_size <= curr_thread_size) {
          slice_end += curr_thread_node_size;
          thread_info->nodes_count[node_id] = 0;
          curr_thread_size -= curr_thread_node_size;
          thread_info = tm.threads_cbuffer.NextItem();
        } else {
          slice_end += curr_thread_size;
          thread_info->nodes_count[node_id] -= curr_thread_size;
          nodes_count_range->begin += curr_thread_size;
          curr_thread_size = 0;
        }
        tm.threads[i].addr.push_back({slice_addr, slice_begin, slice_end});
      }
    }
    curr_thread_size = std::min(block_size,
                                summ_size > block_size*(i+1) ?
                                summ_size - block_size*(i+1) : 0);
  }
}

template <>
void OptPartitionBuilder::UpdateThreadsWork<false, false>(
                  const std::vector<uint16_t>& compleate_trees_depth_wise) {
  uint32_t block_size = common::GetBlockSize(summ_size, n_threads);
  tm.threads_sbuffer.Reset();
  auto thread_info = tm.threads_sbuffer.GetItem();
  uint32_t curr_vec_rowssize = thread_info->vec_rows[0];
  uint32_t curr_thread_size = block_size;
  for (uint32_t i = 0; i < n_threads; ++i) {
    std::vector<uint32_t> borrowed_work;
    while (curr_thread_size != 0) {
      borrowed_work.push_back(tm.threads_sbuffer.Id());
      uint32_t* slice_addr = thread_info->vec_rows.data();
      uint32_t slice_begin = 1 + thread_info->vec_rows[0] - curr_vec_rowssize;
      uint32_t slice_end = 1 + thread_info->vec_rows[0];
      if (curr_vec_rowssize > curr_thread_size) {
        slice_end += -curr_vec_rowssize + curr_thread_size;
        curr_vec_rowssize -= curr_thread_size;
        curr_thread_size = 0;
      } else {
        curr_thread_size -= curr_vec_rowssize;
        thread_info = tm.threads_sbuffer.NextItem();
        curr_vec_rowssize = thread_info->vec_rows[0];
      }
      tm.threads[i].addr.push_back({slice_addr, slice_begin, slice_end});
    }
    curr_thread_size = std::min(block_size,
                                summ_size > block_size*(i+1) ?
                                summ_size - block_size*(i+1) : 0);
    for (const auto& borrowed_tid : borrowed_work) {
      for (const auto& node_id : compleate_trees_depth_wise) {
        if (tm.threads[borrowed_tid].nodes_count[node_id] != 0) {
          tm.Tie(i, node_id);
        }
      }
    }
  }
}

template <>
void OptPartitionBuilder::UpdateThreadsWork<false>(
                  const std::vector<uint16_t>& compleate_trees_depth_wise,
                  GHistIndexMatrix const& gmat, size_t n_features, size_t depth,
                  bool is_left_small, bool check_is_left_small) {
  std::for_each(tm.threads.begin(), tm.threads.end(), [](auto& ti) {ti.addr.clear();});
  if (NeedsBufferUpdate(gmat, n_features, depth)) {
    this->template UpdateThreadsWork<false, true>(compleate_trees_depth_wise);
  } else {
    this->template UpdateThreadsWork<false, false>(compleate_trees_depth_wise);
  }
  std::for_each(tm.threads.begin(), tm.threads.end(), [](auto& ti) {ti.nodes_count.clear();});
  std::for_each(tm.threads.begin(), tm.threads.end(),
                [](auto& ti) {ti.nodes_count_range.clear();});
}

}  // namespace common
}  // namespace xgboost
