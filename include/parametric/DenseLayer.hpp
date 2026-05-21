#pragma once
#include "core/Layer.hpp"

namespace mlengine::parametric {

class DenseLayer : public core::Layer {
 public:
  DenseLayer(int input_dim, int output_dim);

  void forward(const core::MatrixRM& input, core::MatrixRM& output) override;
  core::MatrixRM backward(const core::MatrixRM& output_gradient) override;
  void update_weights(double learning_rate) override;

  core::MatrixRM get_weights() const { return weights_; }
  core::MatrixRM get_bias() const { return bias_; }

 private:
  core::MatrixRM weights_;
  core::MatrixRM bias_;
  core::MatrixRM last_input_;
  core::MatrixRM dW_;
  core::MatrixRM db_;
};

}