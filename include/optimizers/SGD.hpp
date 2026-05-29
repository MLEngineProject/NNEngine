#pragma once

#include "core/Optimizer.hpp"

namespace mlengine::core {

class SGD : public Optimizer {
 public:
  explicit SGD(float learning_rate = 0.01f);

  void step() override;
};

}  // namespace mlengine::core
