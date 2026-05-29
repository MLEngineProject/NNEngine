#pragma once
#include <Eigen/Core>
#include <utility>

#include "core/Types.hpp"

namespace mlengine::autograd {

using MatrixRM = mlengine::MatrixRM;

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

  void update_data(const Eigen::Ref<const MatrixRM>& new_data) {
    if (data.rows() != new_data.rows() || data.cols() != new_data.cols()) {
      data.resize(new_data.rows(), new_data.cols());
    }
    data.noalias() = new_data;
  }

  void zero_grad() {
    if (requires_grad) {
      grad.setZero();
    }
  }
};

}  // namespace mlengine::autograd