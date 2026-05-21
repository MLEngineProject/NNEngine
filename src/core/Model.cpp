#include "core/Model.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace mlengine::core {

Model::Model() { network_ = std::make_shared<parametric::Sequential>(); }

void Model::add(std::shared_ptr<Layer> layer) { network_->add(layer); }

void Model::compile(std::shared_ptr<Loss> loss_fn) { loss_fn_ = loss_fn; }

void Model::fit(const MatrixRM& X, const MatrixRM& y, int epochs,
                double learning_rate, bool verbose) {
  if (!loss_fn_) {
    throw std::runtime_error(
        "Model must be compiled with a loss function before fitting.");
  }

  for (int epoch = 0; epoch < epochs; ++epoch) {
    MatrixRM predictions;
    network_->forward(X, predictions);

    double loss_val = loss_fn_->calculate(predictions, y);

    MatrixRM loss_gradient = loss_fn_->backward(predictions, y);
    network_->backward(loss_gradient);

    network_->update_weights(learning_rate);

    if (verbose &&
        (epoch % std::max(1, epochs / 10) == 0 || epoch == epochs - 1)) {
      std::cout << "Epoch " << epoch << " | Loss: " << loss_val << std::endl;
    }
  }
}

MatrixRM Model::predict(const MatrixRM& X) {
  MatrixRM predictions;
  network_->forward(X, predictions);
  return predictions;
}

}