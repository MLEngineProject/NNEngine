#include "autograd/ops/AddBiasOp.hpp"

namespace mlengine::autograd::ops {

void AddBiasOp::forward() {
  if (out_->data.rows() != a_->data.rows() ||
      out_->data.cols() != a_->data.cols()) {
    out_->data.resize(a_->data.rows(), a_->data.cols());
    if (out_->requires_grad) {
      out_->grad.resize(a_->data.rows(), a_->data.cols());
      out_->grad.setZero();
    }
  }
  out_->data.noalias() = a_->data.rowwise() + b_->data.row(0);
}

void AddBiasOp::backward() {
  if (a_->requires_grad) a_->grad.noalias() += out_->grad;
  if (b_->requires_grad) b_->grad.noalias() += out_->grad.colwise().sum();
}

}  // namespace mlengine::autograd::ops