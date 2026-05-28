#pragma once
#include <cmath>
#include <vector>

#include "autograd/Tensor.hpp"

namespace mlengine::core {

class Optimizer {
 protected:
  std::vector<autograd::Tensor*> parameters_;
  double lr_;

 public:
  Optimizer(double learning_rate) : lr_(learning_rate) {}
  virtual ~Optimizer() = default;

  virtual void set_parameters(const std::vector<autograd::Tensor*>& params) {
    parameters_ = params;
  }

  virtual void step() = 0;

  void zero_grad() {
    for (auto* p : parameters_) {
      p->zero_grad();
    }
  }
};

class SGD : public Optimizer {
 public:
  SGD(double learning_rate = 0.01) : Optimizer(learning_rate) {}

  void step() override {
    for (auto* p : parameters_) {
      if (p->requires_grad) {
        p->data -= lr_ * p->grad;
      }
    }
  }
};

class Adam : public Optimizer {
  std::vector<autograd::MatrixRM> m_;
  std::vector<autograd::MatrixRM> v_;
  int t_ = 0;
  double beta1_ = 0.9;
  double beta2_ = 0.999;
  double epsilon_ = 1e-8;

 public:
  Adam(double learning_rate = 0.001) : Optimizer(learning_rate) {}

  void set_parameters(const std::vector<autograd::Tensor*>& params) override {
    Optimizer::set_parameters(params);
    m_.clear();
    v_.clear();
    t_ = 0;
    for (auto* p : parameters_) {
      m_.push_back(autograd::MatrixRM::Zero(p->data.rows(), p->data.cols()));
      v_.push_back(autograd::MatrixRM::Zero(p->data.rows(), p->data.cols()));
    }
  }

  void step() override {
    t_++;
    for (size_t i = 0; i < parameters_.size(); ++i) {
      auto* p = parameters_[i];
      if (!p->requires_grad) continue;

      m_[i] = beta1_ * m_[i] + (1.0 - beta1_) * p->grad;
      v_[i] = beta2_ * v_[i] + (1.0 - beta2_) * p->grad.cwiseAbs2();

      autograd::MatrixRM m_hat = m_[i] / (1.0 - std::pow(beta1_, t_));
      autograd::MatrixRM v_hat = v_[i] / (1.0 - std::pow(beta2_, t_));

      p->data -= lr_ * m_hat.cwiseQuotient(
                           (v_hat.cwiseSqrt().array() + epsilon_).matrix());
    }
  }
};

}  // namespace mlengine::core