#pragma once
#include "autograd/Tensor.hpp"

namespace mlengine::core {

class Loss {
 public:
  virtual ~Loss() = default;
  virtual double forward(autograd::Tensor* predictions,
                         autograd::Tensor* targets) = 0;
};

class MSELoss : public Loss {
 public:
  double forward(autograd::Tensor* predictions,
                 autograd::Tensor* targets) override {
    if (predictions->requires_grad) {
      predictions->grad =
          2.0 * (predictions->data - targets->data) / predictions->data.rows();
    }
    return (predictions->data - targets->data).squaredNorm() /
           predictions->data.rows();
  }
};

class CategoricalCrossEntropyLoss : public Loss {
 public:
  double forward(autograd::Tensor* predictions,
                 autograd::Tensor* targets) override {
    double epsilon = 1e-15;
    autograd::MatrixRM clipped =
        predictions->data.cwiseMax(epsilon).cwiseMin(1.0 - epsilon);

    if (predictions->requires_grad) {
      predictions->grad = (-(targets->data.array() / clipped.array()) /
                           predictions->data.rows())
                              .matrix();
    }

    return -(targets->data.array() * clipped.array().log()).sum() /
           predictions->data.rows();
  }
};

}  // namespace mlengine::core