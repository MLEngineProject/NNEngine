#pragma once

#include "autograd/Tensor.hpp"
#include "core/Types.hpp"

namespace mlengine::core {

/**
 * @brief Objective function that produces a scalar training signal.
 */
class Loss {
 public:
  virtual ~Loss() = default;
  /**
   * @brief Evaluate the loss and, when appropriate, populate the prediction
   * gradient.
   * @param predictions Model outputs.
   * @param targets Reference targets.
   * @return Scalar loss value for the current batch.
   */
  virtual float forward(autograd::Tensor* predictions,
                        autograd::Tensor* targets) = 0;
};

}  // namespace mlengine::core
