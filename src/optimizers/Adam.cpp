#include "optimizers/Adam.hpp"

#include <cmath>

namespace mlengine::core {

Adam::Adam(float learning_rate) : Optimizer(learning_rate) {}

void Adam::set_parameters(const std::vector<autograd::Tensor*>& params) {
  Optimizer::set_parameters(params);
  m_.clear();
  v_.clear();
  t_ = 0;
  for (auto* p : parameters_) {
    m_.push_back(autograd::MatrixRM::Zero(p->data.rows(), p->data.cols()));
    v_.push_back(autograd::MatrixRM::Zero(p->data.rows(), p->data.cols()));
  }
}

void Adam::step() {
  t_++;
  float current_lr = lr_;

  double bias_corr1 =
      1.0 - std::pow(static_cast<double>(beta1_), static_cast<double>(t_));
  double bias_corr2 =
      1.0 - std::pow(static_cast<double>(beta2_), static_cast<double>(t_));

  for (size_t i = 0; i < parameters_.size(); ++i) {
    auto* p = parameters_[i];
    if (!p->requires_grad) continue;

    float* p_ptr = p->data.data();
    float* g_ptr = p->grad.data();
    float* m_ptr = m_[i].data();
    float* v_ptr = v_[i].data();
    size_t size = p->data.size();

#pragma omp simd
    for (size_t j = 0; j < size; ++j) {
      m_ptr[j] = beta1_ * m_ptr[j] + (1.0f - beta1_) * g_ptr[j];
      v_ptr[j] = beta2_ * v_ptr[j] + (1.0f - beta2_) * (g_ptr[j] * g_ptr[j]);

      p_ptr[j] -=
          current_lr * (m_ptr[j] / static_cast<float>(bias_corr1)) /
          (std::sqrt(v_ptr[j] / static_cast<float>(bias_corr2)) + epsilon_);
    }
  }
}

}  // namespace mlengine::core
