#pragma once

#include "autograd/Tensor.hpp"
#include "core/Types.hpp"

namespace mlengine::core {

/**
 * @brief Base interface for loss functions.
 */
class Loss {
 public:
  virtual ~Loss() = default;
  virtual float forward(autograd::Tensor* predictions,
                        autograd::Tensor* targets) = 0;
};

}  // namespace mlengine::core