#pragma once

#include <memory>
#include <vector>

#include "core/Types.hpp"

namespace mlengine::autograd {
struct Tensor;
}  // namespace mlengine::autograd

namespace mlengine::core {

using MatrixRM = mlengine::MatrixRM;

class Layer;
class Loss;
class Optimizer;
class Regularizer;

}  // namespace mlengine::core

namespace mlengine::layers {
class Sequential;
}  // namespace mlengine::layers

namespace mlengine::core {

/**
 * @brief High-level model wrapper that coordinates layers, losses, optimizers,
 * and regularizers.
 */
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
  std::shared_ptr<layers::Sequential> network_;
  std::shared_ptr<Optimizer> optimizer_;
  std::shared_ptr<Loss> loss_fn_;
  std::shared_ptr<Regularizer> regularizer_;
  std::vector<autograd::Tensor*> parameters_;
};

}  // namespace mlengine::core