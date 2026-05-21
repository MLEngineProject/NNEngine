#pragma once
#include <Eigen/Core>

namespace mlengine::core {

class Loss {
 public:
  virtual ~Loss() = default;
  virtual double calculate(const MatrixRM& predictions,
                           const MatrixRM& targets) = 0;
  virtual MatrixRM backward(const MatrixRM& predictions,
                            const MatrixRM& targets) = 0;
};

class MSELoss : public Loss {
 public:
  double calculate(const MatrixRM& predictions,
                   const MatrixRM& targets) override {
    return (predictions - targets).squaredNorm() / predictions.rows();
  }

  MatrixRM backward(const MatrixRM& predictions,
                    const MatrixRM& targets) override {
    return 2.0 * (predictions - targets) / predictions.rows();
  }
};

class CategoricalCrossEntropyLoss : public Loss {
 public:
  double calculate(const MatrixRM& predictions,
                   const MatrixRM& targets) override {
    double epsilon = 1e-15;
    MatrixRM clipped = predictions.cwiseMax(epsilon).cwiseMin(1.0 - epsilon);
    // L = - (1/N) * sum(Y * log(P))
    return -(targets.array() * clipped.array().log()).sum() /
           predictions.rows();
  }

  MatrixRM backward(const MatrixRM& predictions,
                    const MatrixRM& targets) override {
    double epsilon = 1e-15;
    MatrixRM clipped = predictions.cwiseMax(epsilon).cwiseMin(1.0 - epsilon);
    // dL/dP = - (1/N) * (Y / P)
    return -(targets.array() / clipped.array()) / predictions.rows();
  }
};

}