#include "core/Model.hpp"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>

namespace mlengine::core {

Model::Model() { network_ = std::make_shared<parametric::Sequential>(); }

void Model::add(std::shared_ptr<Layer> layer) { network_->add(layer); }

void Model::compile(std::shared_ptr<Loss> loss_fn) { loss_fn_ = loss_fn; }

void Model::fit(const MatrixRM& X, const MatrixRM& y, int epochs,
                double learning_rate, int batch_size, bool verbose) {
  if (!loss_fn_) {
    throw std::runtime_error(
        "Model must be compiled with a loss function before fitting.");
  }

  int num_samples = X.rows();

  std::vector<int> indices(num_samples);
  std::iota(indices.begin(), indices.end(), 0);

  std::mt19937 rng(42);

  for (int epoch = 0; epoch < epochs; ++epoch) {
    std::shuffle(indices.begin(), indices.end(), rng);

    double epoch_loss = 0.0;
    int num_batches = 0;

    for (int start_idx = 0; start_idx < num_samples; start_idx += batch_size) {
      int current_batch_size = std::min(batch_size, num_samples - start_idx);

      MatrixRM batch_X(current_batch_size, X.cols());
      MatrixRM batch_y(current_batch_size, y.cols());

      for (int i = 0; i < current_batch_size; ++i) {
        batch_X.row(i) = X.row(indices[start_idx + i]);
        batch_y.row(i) = y.row(indices[start_idx + i]);
      }

      MatrixRM predictions;
      network_->forward(batch_X, predictions);

      epoch_loss += loss_fn_->calculate(predictions, batch_y);
      num_batches++;

      MatrixRM loss_gradient = loss_fn_->backward(predictions, batch_y);
      network_->backward(loss_gradient);
      network_->update_weights(learning_rate);
    }

    double avg_epoch_loss = epoch_loss / num_batches;
    if (verbose &&
        (epoch % std::max(1, epochs / 10) == 0 || epoch == epochs - 1)) {
      std::cout << "Epoch " << epoch << " | Loss: " << avg_epoch_loss
                << std::endl;
    }
  }
}

MatrixRM Model::predict(const MatrixRM& X) {
  MatrixRM predictions;
  network_->forward(X, predictions);
  return predictions;
}

}  // namespace mlengine::core