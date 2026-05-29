#pragma once

#include <vector>

#include "autograd/Tensor.hpp"

namespace mlengine::core {

/**
 * @brief Base class for parameter update rules.
 */
class Optimizer {
 protected:
  std::vector<autograd::Tensor*> parameters_;
  float lr_;

 public:
  /**
   * @brief Construct an optimizer with the supplied learning rate.
   * @param learning_rate Step size used by the update rule.
   */
  explicit Optimizer(float learning_rate) : lr_(learning_rate) {}
  virtual ~Optimizer() = default;

  /**
   * @brief Bind the optimizer to the trainable tensors of a model.
   * @param params Mutable parameter pointers to update in place.
   */
  virtual void set_parameters(const std::vector<autograd::Tensor*>& params) {
    parameters_ = params;
  }

  /**
   * @brief Apply one optimization step to the registered parameters.
   */
  virtual void step() = 0;

  /**
   * @brief Zero all registered parameter gradients.
   */
  void zero_grad() {
    for (auto* p : parameters_) {
      p->zero_grad();
    }
  }
};

}  // namespace mlengine::core
