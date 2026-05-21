#pragma once
#include <Eigen/Core>
#include <iostream>

namespace mlengine::core {

using MatrixRM =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

class Layer {
 public:
  virtual ~Layer() = default;

  virtual void forward(const MatrixRM& input, MatrixRM& output) = 0;

  virtual MatrixRM backward(const MatrixRM& output_gradient) = 0;

  virtual void update_weights(double learning_rate) {}
};

}