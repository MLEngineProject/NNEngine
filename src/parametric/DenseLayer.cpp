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

std::vector<autograd::Tensor*> DenseLayer::parameters() {
  return {&weights_, &bias_};
}

}  // namespace mlengine::parametric