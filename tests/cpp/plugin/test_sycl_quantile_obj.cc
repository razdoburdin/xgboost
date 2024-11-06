/**
 * Copyright 2024 by XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/context.h>

#include "../helpers.h"
#include "../objective/test_quantile_obj.h"

namespace xgboost {
TEST(SyclObjective, DeclareUnifiedTest(Quantile)) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});
  TestQuantile(&ctx);
}

TEST(SyclObjective, DeclareUnifiedTest(QuantileIntercept)) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});
  TestQuantileIntercept(&ctx);
}
}  // namespace xgboost
