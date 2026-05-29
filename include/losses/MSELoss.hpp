#pragma once

#include "autograd/Tensor.hpp"
#include "core/Loss.hpp"

namespace mlengine::core {

/**
 * @brief Mean-squared-error objective for regression.
 */
class MSELoss : public Loss {
 public:
  /**
   * @brief Compute the batch MSE and seed the prediction gradient.
   * @param predictions Model outputs.
   * @param targets Regression targets.
   * @return Average squared error over the batch.
   */
  float forward(autograd::Tensor* predictions,
                autograd::Tensor* targets) override;
};

}  // namespace mlengine::core
