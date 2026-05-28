#include "parametric/DenseLayer.hpp"

#include <cmath>
#include <random>

#include "core/Random.hpp"

namespace mlengine::parametric {

DenseLayer::DenseLayer(int input_dim, int output_dim)
    : weights_(core::MatrixRM::Zero(input_dim, output_dim), true, true),
      bias_(core::MatrixRM::Zero(1, output_dim), true, false) {
  std::normal_distribution<double> dist(0.0, std::sqrt(2.0 / input_dim));
  auto& gen = core::rng();

  for (int i = 0; i < weights_.data.rows(); ++i) {
    for (int j = 0; j < weights_.data.cols(); ++j) {
      weights_.data(i, j) = dist(gen);
    }
  }
}

autograd::Tensor* DenseLayer::forward(autograd::Tape& tape,
                                      autograd::Tensor* input) {
  auto* mm = tape.matmul(input, &weights_);
  return tape.add_bias(mm, &bias_);
}

std::vector<autograd::Tensor*> DenseLayer::parameters() {
  return {&weights_, &bias_};
}

}  // namespace mlengine::parametric