#include "core/Model.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "autograd/Tape.hpp"
#include "core/Random.hpp"

namespace mlengine::core {

Model::Model() { network_ = std::make_shared<parametric::Sequential>(); }

void Model::add(std::shared_ptr<Layer> layer) {
  network_->add(layer);
  if (optimizer_) {
    parameters_ = network_->parameters();
    optimizer_->set_parameters(parameters_);
  }
}

void Model::compile(std::shared_ptr<Optimizer> optimizer,
                    std::shared_ptr<Loss> loss_fn,
                    std::shared_ptr<Regularizer> regularizer) {
  if (!optimizer || !loss_fn) {
    throw std::invalid_argument(
        "Optimizer and loss function must be provided.");
  }
  optimizer_ = optimizer;
  loss_fn_ = loss_fn;
  regularizer_ = regularizer;
  parameters_ = network_->parameters();
  optimizer_->set_parameters(parameters_);
}

void Model::fit(Eigen::Ref<const MatrixRM> X, Eigen::Ref<const MatrixRM> y,
                int epochs, int batch_size, float tol, int n_iter_no_change,
                bool verbose) {
  if (!loss_fn_ || !optimizer_) {
    throw std::runtime_error("Model must be compiled before fitting.");
  }
  if (X.rows() != y.rows()) {
    throw std::invalid_argument("Number of samples in X and y must match.");
  }

  int num_samples = X.rows();
  autograd::Tape tape;

  float best_loss = std::numeric_limits<float>::infinity();
  int no_improvement_count = 0;

  std::vector<int> indices(num_samples);
  std::iota(indices.begin(), indices.end(), 0);

  for (int epoch = 0; epoch < epochs; ++epoch) {
    std::shuffle(indices.begin(), indices.end(), core::rng());

    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(num_samples);
    perm.indices() = Eigen::Map<Eigen::VectorXi>(indices.data(), num_samples);

    MatrixRM X_shuffled = perm * X;
    MatrixRM y_shuffled = perm * y;

    float epoch_loss = 0.0f;
    int num_batches = 0;

    for (int start_idx = 0; start_idx < num_samples; start_idx += batch_size) {
      int current_batch_size = std::min(batch_size, num_samples - start_idx);

      autograd::Tensor* X_tensor = tape.push_expr(
          X_shuffled.middleRows(start_idx, current_batch_size), false);
      autograd::Tensor* y_tensor = tape.push_expr(
          y_shuffled.middleRows(start_idx, current_batch_size), false);

      autograd::Tensor* predictions = network_->forward(tape, X_tensor);
      epoch_loss += loss_fn_->forward(predictions, y_tensor);
      num_batches++;

      tape.backward();
      if (regularizer_) epoch_loss += regularizer_->apply(parameters_);
      optimizer_->step();
      optimizer_->zero_grad();
      tape.reset();
    }

    float avg_epoch_loss = epoch_loss / static_cast<float>(num_batches);
    if (verbose &&
        (epoch % std::max(1, epochs / 10) == 0 || epoch == epochs - 1)) {
      std::cout << "Epoch " << epoch << " | Loss: " << avg_epoch_loss
                << std::endl;
    }

    if (best_loss - avg_epoch_loss > tol) {
      best_loss = avg_epoch_loss;
      no_improvement_count = 0;
    } else if (++no_improvement_count >= n_iter_no_change)
      break;
  }
}

MatrixRM Model::predict(Eigen::Ref<const MatrixRM> X) {
  autograd::Tape tape(false);
  autograd::Tensor* X_tensor = tape.push_tensor(X, false);
  autograd::Tensor* predictions = network_->forward(tape, X_tensor);
  return predictions->data;
}

}  // namespace mlengine::core