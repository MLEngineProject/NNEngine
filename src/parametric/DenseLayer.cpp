#include "parametric/DenseLayer.hpp"

#include <cmath>

namespace mlengine::parametric {

DenseLayer::DenseLayer(int input_dim, int output_dim)
    : weights_(core::MatrixRM::Random(input_dim, output_dim) *
                   std::sqrt(6.0 / (input_dim + output_dim)),
               true),
      bias_(core::MatrixRM::Zero(1, output_dim), true) {}

autograd::Tensor* DenseLayer::forward(autograd::Tape& tape,
                                      autograd::Tensor* input) {
  auto* mm = tape.matmul(input, &weights_);
  return tape.add_bias(mm, &bias_);
}

void DenseLayer::update_weights(double learning_rate) {
  weights_.data -= learning_rate * weights_.grad;
  bias_.data -= learning_rate * bias_.grad;
  weights_.zero_grad();
  bias_.zero_grad();
}

}  // namespace mlengine::parametric