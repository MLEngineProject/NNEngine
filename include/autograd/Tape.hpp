#pragma once
#include <deque>
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

struct LeakyReLUOp : public OpNode {
  Tensor *a_, *out_;
  double alpha_;
  LeakyReLUOp(Tensor* a, Tensor* out, double alpha)
      : a_(a), out_(out), alpha_(alpha) {}
  void backward() override {
    if (a_->requires_grad) {
      a_->grad.array() +=
          (a_->data.array() > 0.0).cast<double>() * out_->grad.array() +
          (a_->data.array() <= 0.0).cast<double>() * alpha_ *
              out_->grad.array();
    }
  }
};

struct SoftmaxOp : public OpNode {
  Tensor *a_, *out_;
  SoftmaxOp(Tensor* a, Tensor* out) : a_(a), out_(out) {}
  void backward() override {
    if (a_->requires_grad) {
      Eigen::MatrixXd dot_prod =
          (out_->data.array() * out_->grad.array()).rowwise().sum();
      a_->grad.noalias() += (out_->data.array() *
                             (out_->grad.array() -
                              dot_prod.replicate(1, out_->grad.cols()).array()))
                                .matrix();
    }
  }
};

class Tape {
  std::deque<Tensor> tensor_pool_;
  size_t tensor_idx_ = 0;
  std::vector<std::unique_ptr<OpNode>> ops_;
  bool record_ops_;

 public:
  explicit Tape(bool record_ops = true) : record_ops_(record_ops) {
    ops_.reserve(10000);
  }

  template <typename Derived>
  Tensor* push_expr(const Eigen::MatrixBase<Derived>& expr,
                    bool requires_grad = true) {
    if (tensor_idx_ >= tensor_pool_.size()) {
      tensor_pool_.emplace_back(expr.eval(), record_ops_ && requires_grad);
    } else {
      tensor_pool_[tensor_idx_].data = expr;
      tensor_pool_[tensor_idx_].requires_grad = record_ops_ && requires_grad;
      if (record_ops_ && requires_grad)
        tensor_pool_[tensor_idx_].grad.setZero(expr.rows(), expr.cols());
    }
    return &tensor_pool_[tensor_idx_++];
  }

  Tensor* push_tensor(const MatrixRM& data, bool requires_grad = true) {
    return push_expr(data, requires_grad);
  }

  Tensor* push_tensor(MatrixRM&& data, bool requires_grad = true) {
    if (tensor_idx_ >= tensor_pool_.size()) {
      tensor_pool_.emplace_back(std::move(data), record_ops_ && requires_grad);
    } else {
      tensor_pool_[tensor_idx_].data = std::move(data);
      tensor_pool_[tensor_idx_].requires_grad = record_ops_ && requires_grad;
      if (record_ops_ && requires_grad) {
        tensor_pool_[tensor_idx_].grad.setZero(
            tensor_pool_[tensor_idx_].data.rows(),
            tensor_pool_[tensor_idx_].data.cols());
      }
    }
    return &tensor_pool_[tensor_idx_++];
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

  Tensor* leaky_relu(Tensor* a, double alpha) {
    MatrixRM out_data = a->data.unaryExpr(
        [alpha](double x) { return x > 0.0 ? x : alpha * x; });
    bool req_grad = record_ops_ && a->requires_grad;
    Tensor* out = push_tensor(std::move(out_data), req_grad);
    if (req_grad) ops_.push_back(std::make_unique<LeakyReLUOp>(a, out, alpha));
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
    for (auto it = ops_.rbegin(); it != ops_.rend(); ++it) (*it)->backward();
  }

  void reset() {
    ops_.clear();
    tensor_idx_ = 0;
  }
};

}  // namespace mlengine::autograd