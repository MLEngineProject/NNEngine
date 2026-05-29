#pragma once

#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "autograd/Tape.hpp"
#include "autograd/Tensor.hpp"
#include "core/DataLoader.hpp"
#include "core/Layer.hpp"
#include "core/Loss.hpp"
#include "core/Optimizer.hpp"
#include "core/Regularizer.hpp"
#include "core/Types.hpp"

namespace mlengine::core {

class JITGraph {
 private:
  std::shared_ptr<Layer> model_;
  std::shared_ptr<Optimizer> optimizer_;
  std::shared_ptr<Loss> loss_fn_;
  std::shared_ptr<Regularizer> regularizer_;

  std::shared_ptr<autograd::Tape> tape_;
  autograd::Tensor* X_input_ = nullptr;
  autograd::Tensor* y_input_ = nullptr;
  autograd::Tensor* predictions_ = nullptr;
  std::vector<autograd::Tensor*> parameters_;

 public:
  JITGraph(std::shared_ptr<Layer> model, std::shared_ptr<Optimizer> optimizer,
           std::shared_ptr<Loss> loss_fn,
           std::shared_ptr<Regularizer> regularizer = nullptr)
      : model_(model),
        optimizer_(optimizer),
        loss_fn_(loss_fn),
        regularizer_(regularizer) {}

  float trace_batch(DataLoader& dataloader) {
    if (!dataloader.has_next()) return 0.0f;

    tape_ = std::make_shared<autograd::Tape>(true);
    MatrixRM X_batch, y_batch;
    dataloader.next_batch(X_batch, y_batch);

    X_input_ = tape_->push_tensor(X_batch, false);
    y_input_ = tape_->push_tensor(y_batch, false);

    parameters_ = model_->parameters();
    optimizer_->set_parameters(parameters_);

    predictions_ = model_->forward(tape_.get(), X_input_);

    float loss = loss_fn_->forward(predictions_, y_input_);
    tape_->backward();

    if (regularizer_) loss += regularizer_->apply(parameters_);
    optimizer_->step();

    return loss;
  }

  std::pair<float, size_t> fast_loop(DataLoader& dataloader) {
    float total_loss = 0.0f;
    size_t batch_count = 0;

    while (dataloader.has_next()) {
      dataloader.next_batch(X_input_->data, y_input_->data);

      optimizer_->zero_grad();
      tape_->zero_grads();

      tape_->replay_forward();
      float loss = loss_fn_->forward(predictions_, y_input_);
      tape_->replay_backward();

      if (regularizer_) loss += regularizer_->apply(parameters_);
      optimizer_->step();

      total_loss += loss;
      batch_count++;
    }
    return {total_loss, batch_count};
  }

  void fast_fit(DataLoader& dataloader, int epochs, float tol = 1e-4f,
                int n_iter_no_change = 10, bool verbose = true) {
    float best_loss = std::numeric_limits<float>::infinity();
    int no_improvement_count = 0;

    for (int epoch = 0; epoch < epochs; ++epoch) {
      dataloader.reset();
      auto [total_loss, batches] = fast_loop(dataloader);

      float avg_loss =
          total_loss / static_cast<float>(std::max(size_t(1), batches));

      if (verbose &&
          (epoch % std::max(1, epochs / 10) == 0 || epoch == epochs - 1)) {
        std::cout << "Epoch " << epoch << " | Loss: " << avg_loss << std::endl;
      }

      if (best_loss - avg_loss > tol) {
        best_loss = avg_loss;
        no_improvement_count = 0;
      } else if (++no_improvement_count >= n_iter_no_change) {
        break;
      }
    }
  }
};

}  // namespace mlengine::core