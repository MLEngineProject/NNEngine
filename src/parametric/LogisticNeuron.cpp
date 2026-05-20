#include "parametric/LogisticNeuron.hpp"

#include <cmath>

namespace mlengine::parametric {

void LogisticNeuron::fit(const MatrixRM& X, const VectorRM& y, int epochs,
                         double learning_rate) {
  int m = X.rows();
  int n = X.cols();

  weights_ = VectorRM::Zero(n);
  bias_ = 0.0;

  for (int epoch = 0; epoch < epochs; ++epoch) {
    VectorRM Z = (X * weights_).array() + bias_;

    VectorRM A =
        Z.unaryExpr([](double z) { return 1.0 / (1.0 + std::exp(-z)); });

    VectorRM error = A - y;

    // dW = (X^T * error) / m
    VectorRM dW = (X.transpose() * error) / m;
    double db = error.sum() / m;
    weights_ -= learning_rate * dW;
    bias_ -= learning_rate * db;
  }
}

VectorRM LogisticNeuron::predict_proba(const MatrixRM& X) const {
  VectorRM Z = (X * weights_).array() + bias_;
  return Z.unaryExpr([](double z) { return 1.0 / (1.0 + std::exp(-z)); });
}

VectorRM LogisticNeuron::predict(const MatrixRM& X) const {
  VectorRM proba = predict_proba(X);
  return proba.unaryExpr([](double p) { return p >= 0.5 ? 1.0 : 0.0; });
}

}  // namespace mlengine::parametric