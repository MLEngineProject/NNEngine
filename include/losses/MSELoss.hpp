#pragma once

#include "autograd/Tensor.hpp"
#include "core/Loss.hpp"

namespace mlengine::core {

class MSELoss : public Loss {
 public:
  float forward(autograd::Tensor* predictions,
                autograd::Tensor* targets) override;
};

}  // namespace mlengine::core
