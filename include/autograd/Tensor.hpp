#pragma once
#include <Eigen/Core>

namespace mlengine::autograd {

using MatrixRM =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct Tensor {
  MatrixRM data;
  MatrixRM grad;
  bool requires_grad;

  Tensor(const MatrixRM& val, bool req_grad = true)
      : data(val), requires_grad(req_grad) {
    if (requires_grad) {
      grad = MatrixRM::Zero(val.rows(), val.cols());
    }
  }

  Tensor(MatrixRM&& val, bool req_grad = true)
      : data(std::move(val)), requires_grad(req_grad) {
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