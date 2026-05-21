#include "parametric/ReLULayer.hpp"

namespace mlengine::parametric {

void ReLULayer::forward(const core::MatrixRM& input, core::MatrixRM& output) {
  last_input_ = input;
  output = input.cwiseMax(0.0);
}

core::MatrixRM ReLULayer::backward(const core::MatrixRM& output_gradient) {
  core::MatrixRM dX = output_gradient;
  dX = (last_input_.array() > 0.0).select(dX, 0.0);
  return dX;
}

}