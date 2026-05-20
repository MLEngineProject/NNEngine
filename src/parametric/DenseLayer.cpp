#include "parametric/DenseLayer.hpp"

namespace mlengine::parametric {

DenseLayer::DenseLayer(int input_dim, int output_dim) {
  weights_ = core::MatrixRM::Random(input_dim, output_dim);
  bias_ = core::MatrixRM::Zero(1, output_dim);
}

void DenseLayer::forward(const core::MatrixRM& input, core::MatrixRM& output) {
  last_input_ = input;
  output.noalias() = (input * weights_).rowwise() + bias_.row(0);
}

core::MatrixRM DenseLayer::backward(const core::MatrixRM& output_gradient,
                                    double learning_rate) {
  core::MatrixRM dW = last_input_.transpose() * output_gradient;
  core::MatrixRM dX = output_gradient * weights_.transpose();

  weights_ -= learning_rate * dW;
  bias_ -= learning_rate * output_gradient.colwise().sum();

  return dX;
}

}