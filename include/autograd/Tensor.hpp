#pragma once
#include <Eigen/Core>

namespace mlengine::autograd {

using MatrixRM =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct Tensor {
  MatrixRM data;
  MatrixRM grad;
  bool requires_grad;
  bool apply_regularization;

  Tensor(const MatrixRM& val, bool req_grad = true, bool apply_reg = false)
      : data(val), requires_grad(req_grad), apply_regularization(apply_reg) {
    if (requires_grad) {
      grad = MatrixRM::Zero(val.rows(), val.cols());
    }
  }

  Tensor(MatrixRM&& val, bool req_grad = true, bool apply_reg = false)
      : data(std::move(val)),
        requires_grad(req_grad),
        apply_regularization(apply_reg) {
    if (requires_grad) {
      grad = MatrixRM::Zero(data.rows(), data.cols());
    }
  }

  void zero_grad() {
    if (requires_grad) {
      grad.setZero();
    }
  }
};

}  // namespace mlengine::autograd