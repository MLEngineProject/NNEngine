#include "losses/MSELoss.hpp"

namespace mlengine::core {

float MSELoss::forward(autograd::Tensor* predictions,
                       autograd::Tensor* targets) {
  if (predictions->requires_grad) {
    predictions->grad = 2.0f * (predictions->data - targets->data) /
                        static_cast<float>(predictions->data.rows());
  }
  return (predictions->data - targets->data).squaredNorm() /
         static_cast<float>(predictions->data.rows());
}

}  // namespace mlengine::core
