/*!
 * Copyright 2017-2024 by Contributors
 * \file tree_model.h
 */
#ifndef PLUGIN_SYCL_TREE_TREE_MODEL_H_
#define PLUGIN_SYCL_TREE_TREE_MODEL_H_

#include <xgboost/tree_model.h>

#include "data.h"

namespace xgboost {
namespace sycl {

class RegTree {
 public:
  using Node = xgboost::RegTree::Node;

  static constexpr bst_node_t kInvalidNodeId = xgboost::RegTree::kInvalidNodeId;

  /**
   * \brief Expands a leaf node into two additional leaf nodes.
   *
   * \param nid               The node index to expand.
   * \param split_index       Feature index of the split.
   * \param split_value       The split condition.
   * \param default_left      True to default left.
   * \param base_weight       The base weight, before learning rate.
   * \param left_leaf_weight  The left leaf weight for prediction, modified by learning rate.
   * \param right_leaf_weight The right leaf weight for prediction, modified by learning rate.
   * \param loss_change       The loss change.
   * \param sum_hess          The sum hess.
   * \param left_sum          The sum hess of left leaf.
   * \param right_sum         The sum hess of right leaf.
   * \param leaf_right_child  The right child index of leaf, by default kInvalidNodeId,
   *                          some updaters use the right child index of leaf as a marker
   */

  class NodeExpander {
    Node* active_nodes;
    int n_active_nodes;
  
    int AllocNode() {
      return n_active_nodes++;
    }

   public:
    NodeExpander(Node* _active_nodes, int _n_active_nodes) :
      active_nodes(_active_nodes), n_active_nodes(_n_active_nodes) {}

    void ExpandNode(bst_node_t nid, unsigned split_index, bst_float split_value,
                    bool default_left, bst_float base_weight,
                    bst_float left_leaf_weight, bst_float right_leaf_weight,
                    bst_float loss_change, float sum_hess, float left_sum,
                    float right_sum,
                    bst_node_t leaf_right_child = kInvalidNodeId) {
      int pleft = this->AllocNode();
      int pright = this->AllocNode();
      auto &node = active_nodes[nid];

      node.SetLeftChild(pleft);
      node.SetRightChild(pright);
      active_nodes[node.LeftChild()].SetParent(nid, true);
      active_nodes[node.RightChild()].SetParent(nid, false);
      node.SetSplit(split_index, split_value, default_left);

      active_nodes[pleft].SetLeaf(left_leaf_weight, leaf_right_child);
      active_nodes[pright].SetLeaf(right_leaf_weight, leaf_right_child);
    }

  };

  NodeExpander GetNodeExpander() {
    return NodeExpander(active_nodes_.Data(), n_active_nodes);
  }

  class DepthCalculator {
    const Node* active_nodes;

    public:

    DepthCalculator(const Node* _active_nodes) : active_nodes(_active_nodes) {}

    std::int32_t GetDepth(bst_node_t nid) const {
      int depth = 0;
      while (!active_nodes[nid].IsRoot()) {
        ++depth;
        nid = active_nodes[nid].Parent();
      }
      return depth;
    }
  };

  DepthCalculator GetDepthCalculator() {
    return DepthCalculator(active_nodes_.DataConst());
  }

  void PreAllocate(::sycl::queue* qu, size_t n_nodes) {
    active_nodes_.Resize(qu, n_nodes);
    // deleted_nodes_.Resize(qu, n_nodes);
  }

  ::sycl::event PreAllocate(::sycl::queue* qu, size_t n_nodes,
                            const std::vector<::sycl::event>& events_in) {
    ::sycl::event event_out;
    active_nodes_.Resize(qu, n_nodes, events_in, &event_out);
    return event_out;
    // deleted_nodes_.Resize(qu, n_nodes, Node(), event);
  }

  Node* GetNodesPtr() {
    return active_nodes_.Data();
  }

  ::sycl::event Set(::sycl::queue* qu, xgboost::RegTree* p_tree_host,
                    const std::vector<::sycl::event>& events_in) {
    p_tree_host_ = p_tree_host;
    const std::vector<Node>& nodes_host = p_tree_host->GetNodes();
    auto event_out = PreAllocate(qu, nodes_host.size(), events_in);
    Node* node_ptr = active_nodes_.Data();
    event_out = qu->memcpy(node_ptr, nodes_host.data(), nodes_host.size() * sizeof(Node), event_out);
    return event_out;
  }

  ::sycl::event CopyToHost(::sycl::queue* qu, const std::vector<::sycl::event>& events_in) const {
    auto nodes_host = const_cast<std::vector<Node>&>(p_tree_host_->GetNodes());
    nodes_host.resize(n_active_nodes);
    auto event_out = qu->memcpy(nodes_host.data(), active_nodes_.DataConst(),
                                n_active_nodes * sizeof(Node), events_in);
    return event_out;
  }

 private:
  int n_active_nodes;
  xgboost::RegTree* p_tree_host_;
  USMVector<Node, MemoryType::on_device> active_nodes_;
  // bst_node_t n_deleted_nodes;
  // USMVector<Node, MemoryType::on_device> deleted_nodes_;
};


}  // namespace sycl
}  // namespace xgboost

#endif  // PLUGIN_SYCL_TREE_TREE_MODEL_H_
