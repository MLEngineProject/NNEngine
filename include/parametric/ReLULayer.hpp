#pragma once
#include "core/Layer.hpp"

namespace mlengine::parametric {

class ReLULayer : public core::Layer {
 public:
  autograd::Tensor* forward(autograd::Tape& tape,
                            autograd::Tensor* input) override {
    return tape.relu(input);
  }
};

}  // namespace mlengine::parametric