#pragma once

#include "autograd/Tensor.hpp"
#include "core/Loss.hpp"

namespace mlengine::core {

/**
 * @brief Numerically stable softmax cross-entropy for classification.
 */
class SoftmaxCrossEntropyLoss : public Loss {
 public:
  /**
   * @brief Compute cross-entropy on logits using the log-sum-exp trick.
   * @param logits Unnormalized class scores.
   * @param targets One-hot encoded target probabilities.
   * @return Mean cross-entropy loss for the batch.
   */
  float forward(autograd::Tensor* logits, autograd::Tensor* targets) override;
};

}  // namespace mlengine::core
