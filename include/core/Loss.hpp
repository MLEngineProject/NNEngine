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

}