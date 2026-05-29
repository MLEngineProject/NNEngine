#pragma once
#include "core/Layer.hpp"

namespace mlengine::layers {

class LeakyReLULayer : public core::Layer {
 public:
  LeakyReLULayer(float alpha = 0.01f) : alpha_(alpha) {}

  autograd::Tensor* forward(autograd::Tape* tape,
                            autograd::Tensor* input) override {
    return tape->leaky_relu(input, alpha_);
  }

 private:
  float alpha_;
};

}  // namespace mlengine::layers