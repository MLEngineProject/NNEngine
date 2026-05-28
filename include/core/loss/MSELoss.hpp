#pragma once

#include "autograd/Tensor.hpp"
#include "core/Types.hpp"
#include "core/loss/Loss.hpp"

namespace mlengine::core {

/**
 * @brief Mean squared error loss.
 */
class MSELoss : public Loss {
 public:
  float forward(autograd::Tensor* predictions,
                autograd::Tensor* targets) override {
    if (predictions->requires_grad) {
      predictions->grad = 2.0f * (predictions->data - targets->data) /
                          static_cast<float>(predictions->data.rows());
    }
    return (predictions->data - targets->data).squaredNorm() /
           static_cast<float>(predictions->data.rows());
  }
};

}  // namespace mlengine::core