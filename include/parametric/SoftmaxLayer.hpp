#pragma once
#include "core/Layer.hpp"

namespace mlengine::parametric {

class SoftmaxLayer : public core::Layer {
 public:
  void forward(const core::MatrixRM& input, core::MatrixRM& output) override;
  core::MatrixRM backward(const core::MatrixRM& output_gradient) override;

 private:
  core::MatrixRM P_;
};

}