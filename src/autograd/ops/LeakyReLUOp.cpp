#include "autograd/ops/LeakyReLUOp.hpp"

namespace mlengine::autograd::ops {

void LeakyReLUOp::forward() {
  if (out_->data.rows() != a_->data.rows() ||
      out_->data.cols() != a_->data.cols()) {
    out_->data.resize(a_->data.rows(), a_->data.cols());
    if (out_->requires_grad) {
      out_->grad.resize(a_->data.rows(), a_->data.cols());
      out_->grad.setZero();
    }
  }
  out_->data.noalias() =
      a_->data.unaryExpr([this](float x) { return x > 0.0f ? x : alpha_ * x; });
}

void LeakyReLUOp::backward() {
  if (a_->requires_grad)
    a_->grad.array() +=
        (a_->data.array() > 0.0f)
            .select(out_->grad.array(), alpha_ * out_->grad.array());
}

}  // namespace mlengine::autograd::ops