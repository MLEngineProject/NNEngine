#pragma once

#include <vector>

#include "autograd/Tensor.hpp"
#include "core/Types.hpp"
#include "core/regularizer/Regularizer.hpp"

namespace mlengine::core {

/**
 * @brief L2 regularizer that adds weight decay and penalty.
 */
class L2Regularizer : public Regularizer {
  float l2_;

 public:
  explicit L2Regularizer(float l2 = 0.0001f) : l2_(l2) {}

  float apply(const std::vector<autograd::Tensor*>& parameters) override {
    float penalty = 0.0f;
    for (auto* p : parameters) {
      if (!p->requires_grad) continue;

      float* p_ptr = p->data.data();
      float* g_ptr = p->grad.data();
      size_t size = p->data.size();

      float local_penalty = 0.0f;

#pragma omp simd reduction(+ : local_penalty)
      for (size_t j = 0; j < size; ++j) {
        local_penalty += p_ptr[j] * p_ptr[j];
        g_ptr[j] += l2_ * p_ptr[j];
      }
      penalty += 0.5f * l2_ * local_penalty;
    }
    return penalty;
  }
};

}  // namespace mlengine::core