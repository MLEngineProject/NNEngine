#include "parametric/SoftmaxLayer.hpp"

#include <cmath>

namespace mlengine::parametric {

void SoftmaxLayer::forward(const core::MatrixRM& input,
                           core::MatrixRM& output) {
  output.resize(input.rows(), input.cols());

  for (int i = 0; i < input.rows(); ++i) {
    double max_val = input.row(i).maxCoeff();

    core::MatrixRM exp_row = (input.row(i).array() - max_val).exp();

    output.row(i) = exp_row.array() / exp_row.sum();
  }

  P_ = output;
}

core::MatrixRM SoftmaxLayer::backward(const core::MatrixRM& output_gradient) {
  core::MatrixRM dX(output_gradient.rows(), output_gradient.cols());

  // The Softmax Jacobian derivative for a batch
  for (int i = 0; i < output_gradient.rows(); ++i) {
    auto p_row = P_.row(i).array();
    auto grad_row = output_gradient.row(i).array();

    // dZ = P * dL - P * (P dot dL)
    dX.row(i) = p_row * grad_row - p_row * (p_row * grad_row).sum();
  }

  return dX;
}

}  // namespace mlengine::parametric