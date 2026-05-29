#include "losses/SoftmaxCrossEntropyLoss.hpp"

namespace mlengine::core {

float SoftmaxCrossEntropyLoss::forward(autograd::Tensor* logits,
                                       autograd::Tensor* targets) {
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
    logits->grad.array() = logits->grad.array().exp().colwise() / sums.array();
    logits->grad.array() -= targets->data.array();
    logits->grad.array() /= static_cast<float>(batch_size);
  }

  return loss;
}

}  // namespace mlengine::core
