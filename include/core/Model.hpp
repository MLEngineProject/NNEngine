#pragma once
#include <Eigen/Core>
#include <memory>

#include "core/Layer.hpp"
#include "core/Loss.hpp"
#include "parametric/Sequential.hpp"

namespace mlengine::core {

class Model {
 public:
  Model();

  void add(std::shared_ptr<Layer> layer);
  void compile(std::shared_ptr<Loss> loss_fn);

  void fit(const MatrixRM& X, const MatrixRM& y, int epochs,
           double learning_rate, bool verbose = true);

  MatrixRM predict(const MatrixRM& X);

 private:
  std::shared_ptr<parametric::Sequential> network_;
  std::shared_ptr<Loss> loss_fn_;
};

}