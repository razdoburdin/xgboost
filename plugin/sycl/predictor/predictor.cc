/*!
 * Copyright by Contributors 2017-2023
 */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtautological-constant-compare"
#pragma GCC diagnostic ignored "-W#pragma-messages"
#include <rabit/rabit.h>
#pragma GCC diagnostic pop

#include <cstddef>
#include <limits>
#include <mutex>

#include <CL/sycl.hpp>

#include "../data.h"

#include "dmlc/registry.h"

#include "xgboost/tree_model.h"
#include "xgboost/predictor.h"
#include "xgboost/tree_updater.h"
#include "../../../src/common/timer.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtautological-constant-compare"
#include "../../src/data/adapter.h"
#pragma GCC diagnostic pop
#include "../../src/common/math.h"
#include "../../src/gbm/gbtree_model.h"
#include "../../../src/common/timer.h"                 // for Monitor

#include "../device_manager.h"

namespace xgboost {
namespace sycl {
namespace predictor {

DMLC_REGISTRY_FILE_TAG(predictor_sycl);

union NodeValue {
  float leaf_weight;
  float fvalue;
};

class Node {
  int fidx;
  uint8_t is_leaf;
  NodeValue val;

 public:
  explicit Node(const RegTree::Node& n) {
    fidx = n.SplitIndex();
    if (n.DefaultLeft()) {
      fidx |= (1U << 31);
    }

    if (n.IsLeaf()) {
      is_leaf = 1;
      val.leaf_weight = n.LeafValue();
    } else {
      is_leaf = 0;
      val.fvalue = n.SplitCond();
    }
  }

  bool IsLeaf() const { return is_leaf == 1; }

  int GetFidx() const { return fidx & ((1U << 31) - 1U); }

  bool MissingLeft() const { return (fidx >> 31) != 0; }

  float GetFvalue() const { return val.fvalue; }

  float GetWeight() const { return val.leaf_weight; }
};

class DeviceModel {
  void InitNodes(size_t first_node, const std::vector<RegTree::Node>& src_nodes, size_t idx, size_t src_idx) {
    nodes[first_node + idx] = static_cast<Node>(src_nodes[src_idx]);
    if (!src_nodes[src_idx].IsLeaf()) {
      InitNodes(first_node, src_nodes, 2 * idx + 1, src_nodes[src_idx].LeftChild());
      InitNodes(first_node, src_nodes, 2 * idx + 2, src_nodes[src_idx].RightChild());
    }
  }

 public:
  USMVector<Node> nodes;
  USMVector<int> tree_group;
  int n_nodes;
  int n_trees;
  int max_depth;

  void Init(::sycl::queue* qu, const gbm::GBTreeModel& model, size_t tree_begin, size_t tree_end) {
    max_depth = -1;
    for (int tree_idx = tree_begin; tree_idx < tree_end; tree_idx++) {
      if (model.trees[tree_idx]->HasCategoricalSplit()) {
        LOG(FATAL) << "Categorical features are not yet supported by sycl";
      }
      max_depth = std::max(max_depth, model.trees[tree_idx]->MaxDepth(0));
    }
    n_trees = tree_end - tree_begin;
    n_nodes = (1u << (max_depth + 1)) - 1;

    nodes.Resize(qu, n_nodes * n_trees);
    for (int tree_idx = tree_begin; tree_idx < tree_end; tree_idx++) {
      auto& src_nodes = model.trees[tree_idx]->GetNodes();
      size_t first_node = n_nodes * (tree_idx - tree_begin);

      InitNodes(first_node, src_nodes, 0, 0);
    }

    tree_group.Resize(qu, model.tree_info.size());
    for (size_t tree_idx = 0; tree_idx < model.tree_info.size(); tree_idx++)
      tree_group[tree_idx] = model.tree_info[tree_idx];
  }
};

float GetLeafWeight(int max_depth, const Node* nodes, const float* fval_buff, const uint8_t* miss_buff) {
  size_t idx = 0;
  while (!nodes[idx].IsLeaf()) {
    int fidx = nodes[idx].GetFidx();
    if (miss_buff[fidx] == 1) {
      idx = 2 * idx + 1 + (!nodes[idx].MissingLeft());
    } else {
      const float fvalue = fval_buff[fidx];
      idx = 2 * idx + 1 + (fvalue >= nodes[idx].GetFvalue());
    }
  }
  return nodes[idx].GetWeight();
}

float GetLeafWeight(int max_depth, const Node* nodes, const float* fval_buff) {
  size_t idx = 0;
  for (int depth = 0; depth < max_depth; ++depth) {
    if (nodes[idx].IsLeaf()) nodes[idx].GetWeight();
    int fidx = nodes[idx].GetFidx();
    const float fvalue = fval_buff[fidx];
    idx = 2 * idx + 1 + (fvalue >= nodes[idx].GetFvalue());
  }
  return nodes[idx].GetWeight();
}

template <bool any_missing>
void DevicePredictInternal(::sycl::queue* qu,
                           const sycl::DeviceMatrix& dmat,
                           HostDeviceVector<float>* out_preds,
                           const gbm::GBTreeModel& model,
                           size_t tree_begin,
                           size_t tree_end,
                           common::Monitor* monitor) {
  if (tree_end - tree_begin == 0) return;
  if (out_preds->HostVector().size() == 0) return;

  DeviceModel device_model;
  monitor->Start("DeviceModel::Init");
  device_model.Init(qu, model, tree_begin, tree_end);
  monitor->Stop("DeviceModel::Init");

  const Node* nodes = device_model.nodes.DataConst();
  const int max_depth = device_model.max_depth;
  const size_t n_nodes = device_model.n_nodes;
  const int* tree_group = device_model.tree_group.DataConst();
  const size_t* row_ptr = dmat.row_ptr.DataConst();
  const Entry* data = dmat.data.DataConst();
  int num_features = dmat.p_mat->Info().num_col_;
  int num_rows = dmat.row_ptr.Size() - 1;
  int num_group = model.learner_model_param->num_output_group;

  monitor->Start("Allocate");
  USMVector<float,   MemoryType::on_device> fval_buff(qu, num_features * num_rows);
  USMVector<uint8_t, MemoryType::on_device> miss_buff;
  auto* fval_buff_ptr = fval_buff.Data();

  std::vector<::sycl::event> events(1);
  if constexpr (any_missing) {
    miss_buff.Resize(qu, num_features * num_rows, 1, &events[0]);
  }
  auto* miss_buff_ptr = miss_buff.Data();
  monitor->Stop("Allocate");
  monitor->Start("Calc");
  auto& out_preds_vec = out_preds->HostVector();
  ::sycl::buffer<float, 1> out_preds_buf(out_preds_vec.data(), out_preds_vec.size());
  events[0] = qu->submit([&](::sycl::handler& cgh) {
    cgh.depends_on(events[0]);
    auto out_predictions = out_preds_buf.template get_access<::sycl::access::mode::read_write>(cgh);
    cgh.parallel_for<>(::sycl::range<1>(num_rows), [=](::sycl::id<1> pid) {
      int row_idx = pid[0];
      auto* fval_buff_row_ptr = fval_buff_ptr + num_features * row_idx;
      auto* miss_buff_row_ptr = miss_buff_ptr + num_features * row_idx;

      const Entry* first_entry = data + row_ptr[row_idx];
      const Entry* last_entry = data + row_ptr[row_idx + 1];
      for (const Entry* entry = first_entry; entry < last_entry; entry += 1) {
        fval_buff_row_ptr[entry->index] = entry->fvalue;
        if constexpr (any_missing) {
          miss_buff_row_ptr[entry->index] = 0;
        }
      }

      if (num_group == 1) {
        float sum = 0.0;
        for (int tree_idx = tree_begin; tree_idx < tree_end; tree_idx++) {
          const Node* first_node = nodes + n_nodes * (tree_idx - tree_begin);
          if constexpr (any_missing) {
            sum += GetLeafWeight(max_depth, first_node, fval_buff_row_ptr, miss_buff_row_ptr);
          } else {
            sum += GetLeafWeight(max_depth, first_node, fval_buff_row_ptr);
          }
        }
        out_predictions[row_idx] += sum;
      } else {
        for (int tree_idx = tree_begin; tree_idx < tree_end; tree_idx++) {
          const Node* first_node = nodes + n_nodes * (tree_idx - tree_begin);
          int out_prediction_idx = row_idx * num_group + tree_group[tree_idx];
          if constexpr (any_missing) {
            out_predictions[out_prediction_idx] +=
              GetLeafWeight(max_depth, first_node, fval_buff_row_ptr, miss_buff_row_ptr);
          } else {
            out_predictions[out_prediction_idx] +=
              GetLeafWeight(max_depth, first_node, fval_buff_row_ptr);
          }
        }
      }
    });
  });
  qu->wait();
  monitor->Stop("Calc");
}

class Predictor : public xgboost::Predictor {
  mutable common::Monitor monitor;
 public:
  void InitOutPredictions(const MetaInfo& info,
                          HostDeviceVector<bst_float>* out_preds,
                          const gbm::GBTreeModel& model) const override {
    CHECK_NE(model.learner_model_param->num_output_group, 0);
    size_t n = model.learner_model_param->num_output_group * info.num_row_;
    const auto& base_margin = info.base_margin_.Data()->HostVector();
    out_preds->Resize(n);
    std::vector<bst_float>& out_preds_h = out_preds->HostVector();
    if (base_margin.size() == n) {
      CHECK_EQ(out_preds->Size(), n);
      std::copy(base_margin.begin(), base_margin.end(), out_preds_h.begin());
    } else {
      auto base_score = model.learner_model_param->BaseScore(ctx_)(0);
      if (!base_margin.empty()) {
        std::ostringstream oss;
        oss << "Ignoring the base margin, since it has incorrect length. "
            << "The base margin must be an array of length ";
        if (model.learner_model_param->num_output_group > 1) {
          oss << "[num_class] * [number of data points], i.e. "
              << model.learner_model_param->num_output_group << " * " << info.num_row_
              << " = " << n << ". ";
        } else {
          oss << "[number of data points], i.e. " << info.num_row_ << ". ";
        }
        oss << "Instead, all data points will use "
            << "base_score = " << base_score;
        LOG(WARNING) << oss.str();
      }
      std::fill(out_preds_h.begin(), out_preds_h.end(), base_score);
    }
  }

  explicit Predictor(Context const* context) :
      xgboost::Predictor::Predictor{context},
      cpu_predictor(xgboost::Predictor::Create("cpu_predictor", context)) {monitor.Init("SyclPredictor");}

  void PredictBatch(DMatrix *dmat, PredictionCacheEntry *predts,
                    const gbm::GBTreeModel &model, uint32_t tree_begin,
                    uint32_t tree_end = 0) const override {
    ::sycl::queue qu = device_manager.GetQueue(ctx_->Device());
    // TODO(razdoburdin): remove temporary workaround after cache fix
    sycl::DeviceMatrix device_matrix(qu, dmat);

    auto* out_preds = &predts->predictions;
    if (tree_end == 0) {
      tree_end = model.trees.size();
    }

    monitor.Start("DevicePredictInternal");
    if (tree_begin < tree_end) {
      const bool any_missing = !(dmat->IsDense());
      if (any_missing) {
        DevicePredictInternal<true>(&qu, device_matrix, out_preds, model, tree_begin, tree_end, &monitor);
      } else {
        DevicePredictInternal<false>(&qu, device_matrix, out_preds, model, tree_begin, tree_end, &monitor);
      }
    }
    monitor.Stop("DevicePredictInternal");
  }

  bool InplacePredict(std::shared_ptr<DMatrix> p_m,
                      const gbm::GBTreeModel &model, float missing,
                      PredictionCacheEntry *out_preds, uint32_t tree_begin,
                      unsigned tree_end) const override {
    LOG(WARNING) << "InplacePredict is not yet implemented for SYCL. CPU Predictor is used.";
    return cpu_predictor->InplacePredict(p_m, model, missing, out_preds, tree_begin, tree_end);
  }

  void PredictInstance(const SparsePage::Inst& inst,
                       std::vector<bst_float>* out_preds,
                       const gbm::GBTreeModel& model, unsigned ntree_limit,
                       bool is_column_split) const override {
    LOG(WARNING) << "PredictInstance is not yet implemented for SYCL. CPU Predictor is used.";
    cpu_predictor->PredictInstance(inst, out_preds, model, ntree_limit, is_column_split);
  }

  void PredictLeaf(DMatrix* p_fmat, HostDeviceVector<bst_float>* out_preds,
                   const gbm::GBTreeModel& model, unsigned ntree_limit) const override {
    LOG(WARNING) << "PredictLeaf is not yet implemented for SYCL. CPU Predictor is used.";
    cpu_predictor->PredictLeaf(p_fmat, out_preds, model, ntree_limit);
  }

  void PredictContribution(DMatrix* p_fmat, HostDeviceVector<float>* out_contribs,
                           const gbm::GBTreeModel& model, uint32_t ntree_limit,
                           const std::vector<bst_float>* tree_weights,
                           bool approximate, int condition,
                           unsigned condition_feature) const override {
    LOG(WARNING) << "PredictContribution is not yet implemented for SYCL. CPU Predictor is used.";
    cpu_predictor->PredictContribution(p_fmat, out_contribs, model, ntree_limit, tree_weights,
                                       approximate, condition, condition_feature);
  }

  void PredictInteractionContributions(DMatrix* p_fmat, HostDeviceVector<bst_float>* out_contribs,
                                       const gbm::GBTreeModel& model, unsigned ntree_limit,
                                       const std::vector<bst_float>* tree_weights,
                                       bool approximate) const override {
    LOG(WARNING) << "PredictInteractionContributions is not yet implemented for SYCL. "
                 << "CPU Predictor is used.";
    cpu_predictor->PredictInteractionContributions(p_fmat, out_contribs, model, ntree_limit,
                                                   tree_weights, approximate);
  }

 private:
  DeviceManager device_manager;

  std::unique_ptr<xgboost::Predictor> cpu_predictor;
};

XGBOOST_REGISTER_PREDICTOR(Predictor, "sycl_predictor")
.describe("Make predictions using SYCL.")
.set_body([](Context const* ctx) { return new Predictor(ctx); });

}  // namespace predictor
}  // namespace sycl
}  // namespace xgboost
