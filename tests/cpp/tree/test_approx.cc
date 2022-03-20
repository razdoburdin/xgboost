/*!
 * Copyright 2021-2022, XGBoost contributors.
 */
#include <gtest/gtest.h>

#include "../../../src/tree/updater_approx.h"
#include "../helpers.h"
#include "test_partitioner.h"

namespace xgboost {
namespace tree {
TEST(Approx, Partitioner) {
  size_t n_samples = 1024, n_features = 1, base_rowid = 0;

  auto Xy = RandomDataGenerator{n_samples, n_features, 0}.GenerateDMatrix(true);
  GenericParameter ctx;
  ctx.InitAllowUnknown(Args{});
  std::unordered_map<uint32_t, CPUExpandEntry> candidates;
  candidates[0] = {0, 0, 0.4};

  auto grad = GenerateRandomGradients(n_samples);
  std::vector<float> hess(grad.Size());
  std::transform(grad.HostVector().cbegin(), grad.HostVector().cend(), hess.begin(),
                 [](auto gpair) { return gpair.GetHess(); });

  for (auto const &page : Xy->GetBatches<GHistIndexMatrix>({64, hess, true})) {
    bst_feature_t const split_ind = 0;
    {
      auto min_value = page.cut.MinValues()[split_ind];
      RegTree tree;
      tree.ExpandNode(
          /*nid=*/0, /*split_index=*/0, /*split_value=*/min_value,
          /*default_left=*/true, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
          /*left_sum=*/0.0f,
          /*right_sum=*/0.0f);
      common::ColumnMatrix column_matrix;
      column_matrix.Init(page, 1, 1);
      ApproxRowPartitioner partitioner(&ctx, page, column_matrix, &tree, 2, false);
      candidates[0].split.split_value = min_value;
      candidates[0].split.sindex = 0;
      candidates[0].split.sindex |= (1U << 31);
      std::unordered_map<uint32_t, bool> mask;//(1 << 4, false);
      std::unordered_map<uint32_t, uint16_t> nodes;
      std::vector<uint16_t> cnodes;
      cnodes.resize(1 << (4), 0);
      cnodes[1] = 1;
      // nodes.resize(1 << (4), 0);
      nodes[0] = 1;
      nodes[1] = 2;
      partitioner.UpdatePosition(&ctx, page, &candidates, &tree, &mask, &nodes, cnodes, 0);

      auto const & assignments = partitioner.GetNodeAssignments();
      std::vector<size_t> result(3, 0);
      for (auto node_id : assignments) {
        CHECK_LE(node_id, 2);
        ++result[node_id];
      }
      ASSERT_EQ(result[1], 0);
      ASSERT_EQ(result[2], n_samples);
// >>>>>>> a20b4d1a... partition optimizations
    }
    {
      // ApproxRowPartitioner partitioner{n_samples, base_rowid};
      auto ptr = page.cut.Ptrs()[split_ind + 1];
      float split_value = page.cut.Values().at(ptr / 2);
      RegTree tree;
// <<<<<<< HEAD
//       GetSplit(&tree, split_value, &candidates);
//       auto left_nidx = tree[RegTree::kRoot].LeftChild();
//       partitioner.UpdatePosition(&ctx, page, candidates, &tree);

//       auto elem = partitioner[left_nidx];
//       ASSERT_LT(elem.Size(), n_samples);
//       ASSERT_GT(elem.Size(), 1);
//       for (auto it = elem.begin; it != elem.end; ++it) {
//         auto value = page.cut.Values().at(page.index[*it]);
//         ASSERT_LE(value, split_value);
//       }
//       auto right_nidx = tree[RegTree::kRoot].RightChild();
//       elem = partitioner[right_nidx];
//       for (auto it = elem.begin; it != elem.end; ++it) {
//         auto value = page.cut.Values().at(page.index[*it]);
//         ASSERT_GT(value, split_value) << *it;
// =======
      tree.ExpandNode(
          /*nid=*/RegTree::kRoot, /*split_index=*/split_ind,
          /*split_value=*/split_value,
          /*default_left=*/true, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
          /*left_sum=*/0.0f,
          /*right_sum=*/0.0f);
      common::ColumnMatrix column_matrix;
      column_matrix.Init(page, 1, 1);
      ApproxRowPartitioner partitioner(&ctx, page, column_matrix, &tree, 2, false);
      candidates[0].split.split_value = split_value;
      candidates[0].split.sindex = 0;
      candidates[0].split.sindex |= (1U << 31);
      std::unordered_map<uint32_t, bool> mask;//(1 << 4, false);
      std::unordered_map<uint32_t, uint16_t> nodes;
      std::vector<uint16_t> cnodes;
      cnodes.resize(1 << (4), 0);
      cnodes[1] = 1;
      nodes[0] = 1;
      nodes[1] = 2;
      // partitioner.UpdatePosition(&ctx, page, candidates, &tree, mask, nodes, nodes, 0);
      partitioner.UpdatePosition(&ctx, page, &candidates, &tree, &mask, &nodes, cnodes, 0);
      auto const & assignments = partitioner.GetNodeAssignments();
      size_t row_id = 0;
      for (auto node_id : assignments) {
        if (node_id == 1) { /* left child */
          auto value = page.cut.Values().at(page.index[row_id++]);
          ASSERT_LE(value, split_value);
        } else {            /* right child */
          auto value = page.cut.Values().at(page.index[row_id++]);
          ASSERT_GT(value, split_value);          
        }
// >>>>>>> a20b4d1a... partition optimizations
      }
    }
  }
}
}  // namespace tree
}  // namespace xgboost
