#pragma once
#include <Eigen/Core>
#include <utility>

#include "core/Types.hpp"

namespace mlengine::autograd {

using MatrixRM = mlengine::MatrixRM;

/**
 * @brief Dense value-and-gradient container used by the autograd engine.
 *
 * Tensors store the forward value in row-major Eigen storage and, when
 * gradients are enabled, a matching gradient buffer that is accumulated during
 * backpropagation.
 */
struct Tensor {
  MatrixRM data;
  MatrixRM grad;
  bool requires_grad;
  bool apply_regularization;

  /**
   * @brief Construct a tensor from a matrix copy.
   * @param val Initial tensor value.
   * @param req_grad Whether gradients should be tracked for this tensor.
   * @param apply_reg Whether regularization should be applied to this tensor.
   */
  Tensor(const MatrixRM& val, bool req_grad = true, bool apply_reg = false)
      : data(val), requires_grad(req_grad), apply_regularization(apply_reg) {
    if (requires_grad) {
      grad = MatrixRM::Zero(val.rows(), val.cols());
    }
  }

  /**
   * @brief Construct a tensor by moving a matrix into the forward buffer.
   * @param val Initial tensor value.
   * @param req_grad Whether gradients should be tracked for this tensor.
   * @param apply_reg Whether regularization should be applied to this tensor.
   */
  Tensor(MatrixRM&& val, bool req_grad = true, bool apply_reg = false)
      : data(std::move(val)),
        requires_grad(req_grad),
        apply_regularization(apply_reg) {
    if (requires_grad) {
      grad = MatrixRM::Zero(data.rows(), data.cols());
    }
  }

  /**
   * @brief Replace the forward value while preserving the tensor shape policy.
   * @param new_data New dense value to copy into the tensor.
   */
  void update_data(const Eigen::Ref<const MatrixRM>& new_data) {
    if (data.rows() != new_data.rows() || data.cols() != new_data.cols()) {
      data.resize(new_data.rows(), new_data.cols());
    }
    data.noalias() = new_data;
  }

  /**
   * @brief Zero the gradient buffer when gradient tracking is enabled.
   */
  void zero_grad() {
    if (requires_grad) {
      grad.setZero();
    }
  }
};

}  // namespace mlengine::autograd