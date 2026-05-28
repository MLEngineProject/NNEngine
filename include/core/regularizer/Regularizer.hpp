#pragma once

#include <vector>

#include "autograd/Tensor.hpp"

namespace mlengine::core {

/**
 * @brief Base interface for regularizers.
 */
class Regularizer {
 public:
  virtual ~Regularizer() = default;

  virtual float apply(const std::vector<autograd::Tensor*>& parameters) = 0;
};

}  // namespace mlengine::core