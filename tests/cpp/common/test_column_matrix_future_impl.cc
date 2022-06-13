/*!
 * Copyright 2018-2022 by XGBoost Contributors
 */
#include <dmlc/filesystem.h>
#include <gtest/gtest.h>

#include "../../../src/common/column_matrix_future_impl.h"
#include "../helpers.h"


namespace xgboost {
namespace common {

TEST(DenseColumnFImpl, Test) {
  int32_t max_num_bins[] = {static_cast<int32_t>(std::numeric_limits<uint8_t>::max()) + 1,
                            static_cast<int32_t>(std::numeric_limits<uint16_t>::max()) + 1,
                            static_cast<int32_t>(std::numeric_limits<uint16_t>::max()) + 2};
  for (int32_t max_num_bin : max_num_bins) {
    auto dmat = RandomDataGenerator(100, 10, 0.0).GenerateDMatrix();
    auto sparse_thresh = 0.2;
    GHistIndexMatrix gmat{dmat.get(), max_num_bin, sparse_thresh, false,
                          common::OmpGetNumThreads(0)};
    ColumnMatrixFImpl column_matrix;
    for (auto const& page : dmat->GetBatches<SparsePage>()) {
      column_matrix.Init(page, gmat, sparse_thresh, common::OmpGetNumThreads(0));
    }
    fprintf(stdout, "max_num_bin = %d\tGetTypeSize() = %d\n", max_num_bin, column_matrix.GetTypeSize());

    const auto& column_list = column_matrix.GetColumnList();
    for (auto i = 0ull; i < dmat->Info().num_row_; i++) {
      for (auto j = 0ull; j < dmat->Info().num_col_; j++) {
        const auto& col = column_list[j];
        ASSERT_EQ(gmat.index[i * dmat->Info().num_col_ + j],
                  col->GetGlobalBinIdx(i));
      }
    }
  }
}

inline void CheckSparseColumn(const ColumnFImpl& col_input, const GHistIndexMatrix& gmat) {
  const SparseColumn& col = static_cast<const SparseColumn& >(col_input);
  ASSERT_EQ(col.Size(), gmat.index.Size());
  for (auto i = 0ull; i < col.Size(); i++) {
    ASSERT_EQ(gmat.index[gmat.row_ptr[col.GetRowIdx(i)]],
              col.GetGlobalBinIdx(i));
  }
}

TEST(SparseColumnFImpl, Test) {
  int32_t max_num_bins[] = {static_cast<int32_t>(std::numeric_limits<uint8_t>::max()) + 1,
                            static_cast<int32_t>(std::numeric_limits<uint16_t>::max()) + 1,
                            static_cast<int32_t>(std::numeric_limits<uint16_t>::max()) + 2};
  for (int32_t max_num_bin : max_num_bins) {
    auto dmat = RandomDataGenerator(100, 1, 0.85).GenerateDMatrix();
    GHistIndexMatrix gmat{dmat.get(), max_num_bin, 0.5f, false, common::OmpGetNumThreads(0)};
    ColumnMatrixFImpl column_matrix;
    for (auto const& page : dmat->GetBatches<SparsePage>()) {
      column_matrix.Init(page, gmat, 1.0, common::OmpGetNumThreads(0));
    }
    const auto& column_list = column_matrix.GetColumnList();
    const auto& col = column_list[0];
    CheckSparseColumn(*col, gmat);
  }
}

inline void CheckColumWithMissingValue(const ColumnFImpl& col_input,
                                       const GHistIndexMatrix& gmat) {
  const DenseColumn& col = static_cast<const DenseColumn& >(col_input);
  for (auto i = 0ull; i < col.Size(); i++) {
    if (col.IsMissing(i)) continue;
    EXPECT_EQ(gmat.index[gmat.row_ptr[i]],
              col.GetGlobalBinIdx(i));
  }
}

TEST(DenseColumnWithMissingFImpl, Test) {
  int32_t max_num_bins[] = {static_cast<int32_t>(std::numeric_limits<uint8_t>::max()) + 1,
                            static_cast<int32_t>(std::numeric_limits<uint16_t>::max()) + 1,
                            static_cast<int32_t>(std::numeric_limits<uint16_t>::max()) + 2};
  for (int32_t max_num_bin : max_num_bins) {
    auto dmat = RandomDataGenerator(100, 1, 0.5).GenerateDMatrix();
    GHistIndexMatrix gmat(dmat.get(), max_num_bin, 0.2, false, common::OmpGetNumThreads(0));
    ColumnMatrixFImpl column_matrix;
    for (auto const& page : dmat->GetBatches<SparsePage>()) {
      column_matrix.Init(page, gmat, 0.2, common::OmpGetNumThreads(0));
    }
    const auto& column_list = column_matrix.GetColumnList();
    const auto& col = column_list[0];
    CheckColumWithMissingValue(*col, gmat);
  }
}

void TestGHistIndexMatrixCreationFImpl(size_t nthreads) {
  size_t constexpr kPageSize = 1024, kEntriesPerCol = 3;
  size_t constexpr kEntries = kPageSize * kEntriesPerCol * 2;
  /* This should create multiple sparse pages */
  std::unique_ptr<DMatrix> dmat{CreateSparsePageDMatrix(kEntries)};
  GHistIndexMatrix gmat(dmat.get(), 256, 0.5f, false, common::OmpGetNumThreads(nthreads));
}

TEST(HistIndexCreationWithExternalMemoryFImpl, Test) {
  // Vary the number of threads to make sure that the last batch
  // is distributed properly to the available number of threads
  // in the thread pool
  TestGHistIndexMatrixCreationFImpl(20);
  TestGHistIndexMatrixCreationFImpl(30);
  TestGHistIndexMatrixCreationFImpl(40);
}
}  // namespace common
}  // namespace xgboost