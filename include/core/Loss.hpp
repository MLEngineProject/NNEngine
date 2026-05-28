#pragma once
#include "autograd/Tensor.hpp"

namespace mlengine::core {

class Loss {
 public:
  virtual ~Loss() = default;
  virtual float forward(autograd::Tensor* predictions,
                        autograd::Tensor* targets) = 0;
};

class MSELoss : public Loss {
 public:
  float forward(autograd::Tensor* predictions,
                autograd::Tensor* targets) override {
    if (predictions->requires_grad) {
      predictions->grad = 2.0f * (predictions->data - targets->data) /
                          static_cast<float>(predictions->data.rows());
    }
    return (predictions->data - targets->data).squaredNorm() /
           static_cast<float>(predictions->data.rows());
  }
};

class CategoricalCrossEntropyLoss : public Loss {
 public:
  float forward(autograd::Tensor* predictions,
                autograd::Tensor* targets) override {
    float epsilon = 1e-7f;
    autograd::MatrixRM clipped =
        predictions->data.cwiseMax(epsilon).cwiseMin(1.0f - epsilon);

    if (predictions->requires_grad) {
      predictions->grad = (-(targets->data.array() / clipped.array()) /
                           predictions->data.rows())
                              .matrix();
    }

    return -(targets->data.array() * clipped.array().log()).sum() /
           static_cast<float>(predictions->data.rows());
  }
};

class SoftmaxCrossEntropyLoss : public Loss {
 public:
  float forward(autograd::Tensor* logits, autograd::Tensor* targets) override {
    int batch_size = logits->data.rows();

    logits->grad.noalias() = logits->data;

    Eigen::VectorXf max_vals = logits->grad.rowwise().maxCoeff();
    logits->grad.array().colwise() -= max_vals.array();

    Eigen::VectorXf sums = logits->grad.array().exp().rowwise().sum();

    Eigen::VectorXf log_sums = sums.array().log();
    float loss =
        -(targets->data.array() * (logits->grad.colwise() - log_sums).array())
             .sum() /
        static_cast<float>(batch_size);

    if (logits->requires_grad) {
      logits->grad.array() =
          logits->grad.array().exp().colwise() / sums.array();
      logits->grad.array() -= targets->data.array();
      logits->grad.array() /= static_cast<float>(batch_size);
    }

    return loss;
  }
};

}  // namespace mlengine::core