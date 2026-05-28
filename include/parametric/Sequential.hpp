#pragma once
#include <memory>
#include <vector>

#include "core/Layer.hpp"

namespace mlengine::parametric {

class Sequential : public core::Layer {
 public:
  void add(std::shared_ptr<core::Layer> layer) { layers_.push_back(layer); }

  autograd::Tensor* forward(autograd::Tape& tape,
                            autograd::Tensor* input) override {
    autograd::Tensor* current = input;
    for (auto& layer : layers_) {
      current = layer->forward(tape, current);
    }
    return current;
  }

  void update_weights(double learning_rate) override {
    for (auto& layer : layers_) {
      layer->update_weights(learning_rate);
    }
  }

 private:
  std::vector<std::shared_ptr<core::Layer>> layers_;
};

}  // namespace mlengine::parametric