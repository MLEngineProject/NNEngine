#pragma once
#include <Eigen/Core>
#include <memory>
#include <vector>

#include "core/Layer.hpp"
#include "core/Loss.hpp"
#include "core/Optimizer.hpp"
#include "core/Regularizer.hpp"
#include "parametric/Sequential.hpp"

namespace mlengine::core {

class Model {
 public:
  Model();

  void add(std::shared_ptr<Layer> layer);

  void compile(std::shared_ptr<Optimizer> optimizer,
               std::shared_ptr<Loss> loss_fn,
               std::shared_ptr<Regularizer> regularizer = nullptr);

  void fit(Eigen::Ref<const MatrixRM> X, Eigen::Ref<const MatrixRM> y,
           int epochs, int batch_size = 32, float tol = 1e-4f,
           int n_iter_no_change = 10, bool verbose = true);

  MatrixRM predict(Eigen::Ref<const MatrixRM> X);

 private:
  std::shared_ptr<parametric::Sequential> network_;
  std::shared_ptr<Optimizer> optimizer_;
  std::shared_ptr<Loss> loss_fn_;
  std::shared_ptr<Regularizer> regularizer_;
  std::vector<autograd::Tensor*> parameters_;
};

}  // namespace mlengine::core