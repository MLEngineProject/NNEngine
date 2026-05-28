#pragma once
#include <Eigen/Core>
#include <memory>

#include "core/Layer.hpp"
#include "core/Loss.hpp"
#include "core/Optimizer.hpp"
#include "parametric/Sequential.hpp"

namespace mlengine::core {

class Model {
 public:
  Model();

  void add(std::shared_ptr<Layer> layer);

  void compile(std::shared_ptr<Optimizer> optimizer,
               std::shared_ptr<Loss> loss_fn);

  void fit(const MatrixRM& X, const MatrixRM& y, int epochs,
           int batch_size = 32, double tol = 1e-4, int n_iter_no_change = 10,
           bool verbose = true);

  MatrixRM predict(const MatrixRM& X);

 private:
  std::shared_ptr<parametric::Sequential> network_;
  std::shared_ptr<Optimizer> optimizer_;
  std::shared_ptr<Loss> loss_fn_;
};

}  // namespace mlengine::core