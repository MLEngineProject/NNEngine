#pragma once

#include <deque>
#include <memory>
#include <vector>

#include "autograd/Tensor.hpp"

namespace mlengine::autograd {

enum class OpType { MatMul, AddBias, ReLU, LeakyReLU };

struct OpNode {
  OpType type;
  Tensor *a, *b, *out;
  float alpha;

  void forward() {
    switch (type) {
      case OpType::MatMul:
        if (out->data.rows() != a->data.rows()) {
          out->data.resize(a->data.rows(), b->data.cols());
          if (out->requires_grad) {
            out->grad.resize(a->data.rows(), b->data.cols());
            out->grad.setZero();
          }
        }
        out->data.noalias() = a->data * b->data;
        break;
      case OpType::AddBias:
        if (out->data.rows() != a->data.rows()) {
          out->data.resize(a->data.rows(), a->data.cols());
          if (out->requires_grad) {
            out->grad.resize(a->data.rows(), a->data.cols());
            out->grad.setZero();
          }
        }
        out->data.noalias() = a->data.rowwise() + b->data.row(0);
        break;
      case OpType::ReLU:
        if (out->data.rows() != a->data.rows()) {
          out->data.resize(a->data.rows(), a->data.cols());
          if (out->requires_grad) {
            out->grad.resize(a->data.rows(), a->data.cols());
            out->grad.setZero();
          }
        }
        out->data.noalias() = a->data.cwiseMax(0.0f);
        break;
      case OpType::LeakyReLU:
        if (out->data.rows() != a->data.rows()) {
          out->data.resize(a->data.rows(), a->data.cols());
          if (out->requires_grad) {
            out->grad.resize(a->data.rows(), a->data.cols());
            out->grad.setZero();
          }
        }
        out->data.noalias() = a->data.unaryExpr(
            [this](float x) { return x > 0.0f ? x : alpha * x; });
        break;
    }
  }

  void backward() {
    switch (type) {
      case OpType::MatMul:
        if (a->requires_grad)
          a->grad.noalias() += out->grad * b->data.transpose();
        if (b->requires_grad)
          b->grad.noalias() += a->data.transpose() * out->grad;
        break;
      case OpType::AddBias:
        if (a->requires_grad) a->grad.noalias() += out->grad;
        if (b->requires_grad) b->grad.noalias() += out->grad.colwise().sum();
        break;
      case OpType::ReLU:
        if (a->requires_grad)
          a->grad.array() +=
              (a->data.array() > 0.0f).select(out->grad.array(), 0.0f);
        break;
      case OpType::LeakyReLU:
        if (a->requires_grad)
          a->grad.array() +=
              (a->data.array() > 0.0f)
                  .select(out->grad.array(), alpha * out->grad.array());
        break;
    }
  }
};

class Tape {
 public:
  std::deque<Tensor> tensor_pool_;
  size_t tensor_idx_ = 0;
  std::vector<OpNode> ops_;
  bool record_ops_;

  explicit Tape(bool record_ops = true) : record_ops_(record_ops) {
    ops_.reserve(10000);
  }

  Tensor* alloc_tensor(int rows, int cols, bool requires_grad = true) {
    bool req_grad_actual = record_ops_ && requires_grad;
    if (tensor_idx_ >= tensor_pool_.size()) {
      tensor_pool_.emplace_back(mlengine::MatrixRM(rows, cols),
                                req_grad_actual);
    } else {
      if (tensor_pool_[tensor_idx_].data.rows() != rows ||
          tensor_pool_[tensor_idx_].data.cols() != cols) {
        tensor_pool_[tensor_idx_].data.resize(rows, cols);
      }
      tensor_pool_[tensor_idx_].requires_grad = req_grad_actual;
      if (req_grad_actual) {
        if (tensor_pool_[tensor_idx_].grad.rows() != rows ||
            tensor_pool_[tensor_idx_].grad.cols() != cols) {
          tensor_pool_[tensor_idx_].grad.resize(rows, cols);
        }
        tensor_pool_[tensor_idx_].grad.setZero();
      }
    }
    return &tensor_pool_[tensor_idx_++];
  }

  template <typename Derived>
  Tensor* push_expr(const Eigen::MatrixBase<Derived>& expr,
                    bool requires_grad = true) {
    Tensor* t = alloc_tensor(expr.rows(), expr.cols(), requires_grad);
    t->data.noalias() = expr;
    return t;
  }

  Tensor* push_tensor(const mlengine::MatrixRM& data,
                      bool requires_grad = true) {
    return push_expr(data, requires_grad);
  }

  Tensor* matmul(Tensor* a, Tensor* b) {
    bool req_grad = record_ops_ && (a->requires_grad || b->requires_grad);
    Tensor* out = alloc_tensor(a->data.rows(), b->data.cols(), req_grad);
    out->data.noalias() = a->data * b->data;
    if (record_ops_) ops_.push_back({OpType::MatMul, a, b, out, 0.0f});
    return out;
  }

  Tensor* add_bias(Tensor* a, Tensor* b) {
    bool req_grad = record_ops_ && (a->requires_grad || b->requires_grad);
    Tensor* out = alloc_tensor(a->data.rows(), a->data.cols(), req_grad);
    out->data.noalias() = a->data.rowwise() + b->data.row(0);
    if (record_ops_) ops_.push_back({OpType::AddBias, a, b, out, 0.0f});
    return out;
  }

  Tensor* relu(Tensor* a) {
    bool req_grad = record_ops_ && a->requires_grad;
    Tensor* out = alloc_tensor(a->data.rows(), a->data.cols(), req_grad);
    out->data.noalias() = a->data.cwiseMax(0.0f);
    if (record_ops_) ops_.push_back({OpType::ReLU, a, nullptr, out, 0.0f});
    return out;
  }

  Tensor* leaky_relu(Tensor* a, float alpha) {
    bool req_grad = record_ops_ && a->requires_grad;
    Tensor* out = alloc_tensor(a->data.rows(), a->data.cols(), req_grad);
    out->data.noalias() = a->data.unaryExpr(
        [alpha](float x) { return x > 0.0f ? x : alpha * x; });
    if (record_ops_)
      ops_.push_back({OpType::LeakyReLU, a, nullptr, out, alpha});
    return out;
  }

  void replay_forward() {
    for (auto& op : ops_) op.forward();
  }

  void replay_backward() {
    for (auto it = ops_.rbegin(); it != ops_.rend(); ++it) it->backward();
  }

  void zero_grads() {
    for (auto& t : tensor_pool_) t.zero_grad();
  }

  void backward() { replay_backward(); }

  void reset() {
    ops_.clear();
    tensor_idx_ = 0;
  }
};

}  // namespace mlengine::autograd