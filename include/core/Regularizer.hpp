#pragma once
#include <vector>

#include "autograd/Tensor.hpp"

namespace mlengine::core {

class Regularizer {
 public:
  virtual ~Regularizer() = default;

  virtual double apply(const std::vector<autograd::Tensor*>& parameters) = 0;
};

class L2Regularizer : public Regularizer {
  double l2_;

 public:
  L2Regularizer(double l2 = 0.0001) : l2_(l2) {}

  double apply(const std::vector<autograd::Tensor*>& parameters) override {
    double penalty = 0.0;
    for (auto* p : parameters) {
      if (p->requires_grad && p->apply_regularization) {
        penalty += 0.5 * l2_ * p->data.squaredNorm();
        p->grad += l2_ * p->data;
      }
    }
    return penalty;
  }
};

}  // namespace mlengine::core