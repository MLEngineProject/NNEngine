#pragma once
#include "core/Layer.hpp"

namespace mlengine::parametric {

class LeakyReLULayer : public core::Layer {
 public:
  LeakyReLULayer(double alpha = 0.01) : alpha_(alpha) {}

  autograd::Tensor* forward(autograd::Tape& tape,
                            autograd::Tensor* input) override {
    return tape.leaky_relu(input, alpha_);
  }

 private:
  double alpha_;
};

}  // namespace mlengine::parametric