#pragma once
#include "core/Layer.hpp"

namespace mlengine::parametric {

class SoftmaxLayer : public core::Layer {
 public:
  autograd::Tensor* forward(autograd::Tape& tape,
                            autograd::Tensor* input) override {
    return tape.softmax(input);
  }
};

}  // namespace mlengine::parametric