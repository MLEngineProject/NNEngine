#pragma once

#include <vector>

#include "autograd/Tensor.hpp"
#include "core/Optimizer.hpp"

namespace mlengine::core {

/**
 * @brief Adam optimizer with first/second moment estimates.
 */
class Adam : public Optimizer {
  std::vector<autograd::MatrixRM> m_;
  std::vector<autograd::MatrixRM> v_;
  int t_ = 0;
  float beta1_ = 0.9f;
  float beta2_ = 0.999f;
  float epsilon_ = 1e-8f;

 public:
  /**
   * @brief Construct Adam with the supplied learning rate.
   * @param learning_rate Base step size used for updates.
   */
  explicit Adam(float learning_rate = 0.001f);

  /**
   * @brief Bind parameter tensors and reset moment buffers.
   * @param params Mutable parameter pointers to update in place.
   */
  void set_parameters(const std::vector<autograd::Tensor*>& params) override;

  /**
   * @brief Apply Adam updates with bias correction on both moments.
   */
  void step() override;
};

}  // namespace mlengine::core
