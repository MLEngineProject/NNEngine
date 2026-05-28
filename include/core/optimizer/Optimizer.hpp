#pragma once

#include <vector>

#include "autograd/Tensor.hpp"

namespace mlengine::core {

/**
 * @brief Base interface for optimizers.
 */
class Optimizer {
 protected:
  std::vector<autograd::Tensor*> parameters_;
  float lr_;

 public:
  explicit Optimizer(float learning_rate) : lr_(learning_rate) {}
  virtual ~Optimizer() = default;

  virtual void set_parameters(const std::vector<autograd::Tensor*>& params) {
    parameters_ = params;
  }

  virtual void step() = 0;

  void zero_grad() {
    for (auto* p : parameters_) {
      p->zero_grad();
    }
  }
};

}  // namespace mlengine::core