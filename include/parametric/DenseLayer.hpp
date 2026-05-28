#pragma once
#include <vector>

#include "autograd/Tensor.hpp"
#include "core/Layer.hpp"

namespace mlengine::parametric {

class DenseLayer : public core::Layer {
 public:
  DenseLayer(int input_dim, int output_dim);
  autograd::Tensor* forward(autograd::Tape& tape,
                            autograd::Tensor* input) override;

  std::vector<autograd::Tensor*> parameters() override;

  core::MatrixRM get_weights() const { return weights_.data; }
  core::MatrixRM get_bias() const { return bias_.data; }

 private:
  autograd::Tensor weights_;
  autograd::Tensor bias_;
};

}  // namespace mlengine::parametric