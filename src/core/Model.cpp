#include "core/Model.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

#include "autograd/Tape.hpp"

namespace mlengine::core {

Model::Model() { network_ = std::make_shared<parametric::Sequential>(); }

void Model::add(std::shared_ptr<Layer> layer) { network_->add(layer); }

void Model::compile(std::shared_ptr<Optimizer> optimizer,
                    std::shared_ptr<Loss> loss_fn) {
  optimizer_ = optimizer;
  loss_fn_ = loss_fn;
  optimizer_->set_parameters(network_->parameters());
}

void Model::fit(const MatrixRM& X, const MatrixRM& y, int epochs,
                int batch_size, double tol, int n_iter_no_change,
                bool verbose) {
  if (!loss_fn_ || !optimizer_) {
    throw std::runtime_error(
        "Model must be compiled with an optimizer and loss function before "
        "fitting.");
  }

  int num_samples = X.rows();
  std::mt19937 rng(42);
  autograd::Tape tape;

  double best_loss = std::numeric_limits<double>::infinity();
  int no_improvement_count = 0;

  MatrixRM X_shuffled(X.rows(), X.cols());
  MatrixRM y_shuffled(y.rows(), y.cols());
  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(num_samples);
  perm.indices() = Eigen::VectorXi::LinSpaced(num_samples, 0, num_samples - 1);

  for (int epoch = 0; epoch < epochs; ++epoch) {
    // 1. Shuffle indices
    std::shuffle(perm.indices().data(), perm.indices().data() + num_samples,
                 rng);

    X_shuffled.noalias() = perm * X;
    y_shuffled.noalias() = perm * y;

    double epoch_loss = 0.0;
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

      optimizer_->step();
      optimizer_->zero_grad();

      tape.reset();
    }

    double avg_epoch_loss = epoch_loss / num_batches;

    if (verbose &&
        (epoch % std::max(1, epochs / 10) == 0 || epoch == epochs - 1)) {
      std::cout << "Epoch " << epoch << " | Loss: " << avg_epoch_loss
                << std::endl;
    }

    if (best_loss - avg_epoch_loss > tol) {
      best_loss = avg_epoch_loss;
      no_improvement_count = 0;
    } else {
      no_improvement_count++;
    }

    if (no_improvement_count >= n_iter_no_change) {
      if (verbose) {
        std::cout << "Early stopping at epoch " << epoch
                  << " (Loss did not improve by " << tol << " for "
                  << n_iter_no_change << " consecutive epochs)." << std::endl;
      }
      break;
    }
  }
}

MatrixRM Model::predict(const MatrixRM& X) {
  autograd::Tape tape(false);
  autograd::Tensor* X_tensor = tape.push_tensor(X, false);
  autograd::Tensor* predictions = network_->forward(tape, X_tensor);
  return predictions->data;
}

}  // namespace mlengine::core