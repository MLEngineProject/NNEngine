#pragma once

#include <vector>

#include "autograd/Tensor.hpp"
#include "core/Regularizer.hpp"

namespace mlengine::core {

class L2Regularizer : public Regularizer {
  float l2_;

 public:
  explicit L2Regularizer(float l2 = 0.0001f);

  float apply(const std::vector<autograd::Tensor*>& parameters) override;
};

}  // namespace mlengine::core
