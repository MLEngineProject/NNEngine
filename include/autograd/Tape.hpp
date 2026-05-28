#pragma once
#include <memory>
#include <utility>
#include <vector>

#include "autograd/Tensor.hpp"

namespace mlengine::autograd {

struct OpNode {
  virtual ~OpNode() = default;
  virtual void backward() = 0;
};

struct MatMulOp : public OpNode {
  Tensor *a_, *b_, *out_;
  MatMulOp(Tensor* a, Tensor* b, Tensor* out) : a_(a), b_(b), out_(out) {}

  void backward() override {
    if (a_->requires_grad)
      a_->grad.noalias() += out_->grad * b_->data.transpose();
    if (b_->requires_grad)
      b_->grad.noalias() += a_->data.transpose() * out_->grad;
  }
};

struct AddBiasOp : public OpNode {
  Tensor *a_, *b_, *out_;
  AddBiasOp(Tensor* a, Tensor* b, Tensor* out) : a_(a), b_(b), out_(out) {}

  void backward() override {
    if (a_->requires_grad) a_->grad.noalias() += out_->grad;
    if (b_->requires_grad) b_->grad.noalias() += out_->grad.colwise().sum();
  }
};

struct ReLUOp : public OpNode {
  Tensor *a_, *out_;
  ReLUOp(Tensor* a, Tensor* out) : a_(a), out_(out) {}

  void backward() override {
    if (a_->requires_grad) {
      a_->grad.array() +=
          (a_->data.array() > 0.0).cast<double>() * out_->grad.array();
    }
  }
};

struct SoftmaxOp : public OpNode {
  Tensor *a_, *out_;
  SoftmaxOp(Tensor* a, Tensor* out) : a_(a), out_(out) {}

  void backward() override {
    if (a_->requires_grad) {
      for (int i = 0; i < out_->grad.rows(); ++i) {
        auto p_row = out_->data.row(i).array();
        auto grad_row = out_->grad.row(i).array();
        a_->grad.row(i).array() +=
            p_row * grad_row - p_row * (p_row * grad_row).sum();
      }
    }
  }
};

class Tape {
  std::vector<std::unique_ptr<Tensor>> intermediates_;
  std::vector<std::unique_ptr<OpNode>> ops_;
  bool record_ops_;

 public:
  explicit Tape(bool record_ops = true) : record_ops_(record_ops) {}

  Tensor* push_tensor(const MatrixRM& data, bool requires_grad = true) {
    intermediates_.push_back(
        std::make_unique<Tensor>(data, record_ops_ && requires_grad));
    return intermediates_.back().get();
  }

  Tensor* push_tensor(MatrixRM&& data, bool requires_grad = true) {
    intermediates_.push_back(std::make_unique<Tensor>(
        std::move(data), record_ops_ && requires_grad));
    return intermediates_.back().get();
  }

  Tensor* matmul(Tensor* a, Tensor* b) {
    MatrixRM out_data(a->data.rows(), b->data.cols());
    out_data.noalias() = a->data * b->data;

    bool req_grad = record_ops_ && (a->requires_grad || b->requires_grad);
    Tensor* out = push_tensor(std::move(out_data), req_grad);
    if (req_grad) ops_.push_back(std::make_unique<MatMulOp>(a, b, out));
    return out;
  }

  Tensor* add_bias(Tensor* a, Tensor* b) {
    MatrixRM out_data = a->data.rowwise() + b->data.row(0);

    bool req_grad = record_ops_ && (a->requires_grad || b->requires_grad);
    Tensor* out = push_tensor(std::move(out_data), req_grad);
    if (req_grad) ops_.push_back(std::make_unique<AddBiasOp>(a, b, out));
    return out;
  }

  Tensor* relu(Tensor* a) {
    MatrixRM out_data = a->data.cwiseMax(0.0);

    bool req_grad = record_ops_ && a->requires_grad;
    Tensor* out = push_tensor(std::move(out_data), req_grad);
    if (req_grad) ops_.push_back(std::make_unique<ReLUOp>(a, out));
    return out;
  }

  Tensor* softmax(Tensor* a) {
    MatrixRM out_data(a->data.rows(), a->data.cols());
    for (int i = 0; i < a->data.rows(); ++i) {
      double max_val = a->data.row(i).maxCoeff();
      MatrixRM exp_row = (a->data.row(i).array() - max_val).exp();
      out_data.row(i) = (exp_row.array() / exp_row.sum()).matrix();
    }

    bool req_grad = record_ops_ && a->requires_grad;
    Tensor* out = push_tensor(std::move(out_data), req_grad);
    if (req_grad) ops_.push_back(std::make_unique<SoftmaxOp>(a, out));
    return out;
  }

  void backward() {
    for (auto it = ops_.rbegin(); it != ops_.rend(); ++it) {
      (*it)->backward();
    }
  }

  void reset() {
    ops_.clear();
    intermediates_.clear();
  }
};

}  // namespace mlengine::autograd