#pragma once
#include <Eigen/Core>

#include "autograd/Tape.hpp"

namespace mlengine::core {

using MatrixRM =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

class Layer {
 public:
  virtual ~Layer() = default;

  virtual autograd::Tensor* forward(autograd::Tape& tape,
                                    autograd::Tensor* input) = 0;
  virtual void update_weights(double learning_rate) {}
};

}  // namespace mlengine::core