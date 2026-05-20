#pragma once
#include <Eigen/Core>

namespace mlengine::parametric {

using MatrixRM =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using VectorRM = Eigen::Matrix<double, Eigen::Dynamic, 1>;

class LogisticNeuron {
 public:
  LogisticNeuron() = default;

  void fit(const MatrixRM& X, const VectorRM& y, int epochs,
           double learning_rate);

  VectorRM predict_proba(const MatrixRM& X) const;

  VectorRM predict(const MatrixRM& X) const;

  VectorRM get_weights() const { return weights_; }
  double get_bias() const { return bias_; }

 private:
  VectorRM weights_;
  double bias_ = 0.0;
};

}