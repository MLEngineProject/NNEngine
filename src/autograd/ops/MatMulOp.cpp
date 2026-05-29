#include "autograd/ops/MatMulOp.hpp"

namespace mlengine::autograd::ops {

void MatMulOp::forward() {
  if (out_->data.rows() != a_->data.rows() ||
      out_->data.cols() != b_->data.cols()) {
    out_->data.resize(a_->data.rows(), b_->data.cols());
    if (out_->requires_grad) {
      out_->grad.resize(a_->data.rows(), b_->data.cols());
      out_->grad.setZero();
    }
  }
  out_->data.noalias() = a_->data * b_->data;
}

void MatMulOp::backward() {
  if (a_->requires_grad)
    a_->grad.noalias() += out_->grad * b_->data.transpose();
  if (b_->requires_grad)
    b_->grad.noalias() += a_->data.transpose() * out_->grad;
}

}  // namespace mlengine::autograd::ops