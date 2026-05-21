#include "parametric/DenseLayer.hpp"

#include <cmath>

namespace mlengine::parametric {

DenseLayer::DenseLayer(int input_dim, int output_dim) {
  double limit = std::sqrt(6.0 / (input_dim + output_dim));
  weights_ = core::MatrixRM::Random(input_dim, output_dim) * limit;
  bias_ = core::MatrixRM::Zero(1, output_dim);
}

void DenseLayer::forward(const core::MatrixRM& input, core::MatrixRM& output) {
  last_input_ = input;
  output.noalias() = (input * weights_).rowwise() + bias_.row(0);
}

core::MatrixRM DenseLayer::backward(const core::MatrixRM& output_gradient) {
  dW_.noalias() = last_input_.transpose() * output_gradient;
  db_ = output_gradient.colwise().sum();

  return output_gradient * weights_.transpose();
}

void DenseLayer::update_weights(double learning_rate) {
  weights_ -= learning_rate * dW_;
  bias_ -= learning_rate * db_;
}

}  // namespace mlengine::parametric