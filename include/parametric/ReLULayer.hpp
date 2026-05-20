#pragma once
#include "core/Layer.hpp"

namespace mlengine::parametric {

class ReLULayer : public core::Layer {
 public:
  void forward(const core::MatrixRM& input, core::MatrixRM& output) override;

  core::MatrixRM backward(const core::MatrixRM& output_gradient,
                          double learning_rate) override;

 private:
  core::MatrixRM last_input_;
};

}  // namespace mlengine::parametric