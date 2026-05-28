#pragma once

#include "autograd/Tensor.hpp"
#include "core/Types.hpp"
#include "core/optimizer/Optimizer.hpp"

namespace mlengine::core {

/**
 * @brief Stochastic gradient descent optimizer.
 */
class SGD : public Optimizer {
 public:
  explicit SGD(float learning_rate = 0.01f) : Optimizer(learning_rate) {}

  void step() override {
    for (auto* p : parameters_) {
      if (p->requires_grad) {
        p->data -= lr_ * p->grad;
      }
    }
  }
};

}  // namespace mlengine::core