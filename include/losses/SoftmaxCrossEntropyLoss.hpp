#pragma once

#include "autograd/Tensor.hpp"
#include "core/Loss.hpp"

namespace mlengine::core {

class SoftmaxCrossEntropyLoss : public Loss {
 public:
  float forward(autograd::Tensor* logits, autograd::Tensor* targets) override;
};

}  // namespace mlengine::core
