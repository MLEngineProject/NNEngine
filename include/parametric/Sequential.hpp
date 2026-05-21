#pragma once
#include <iostream>
#include <memory>
#include <vector>

#include "core/Layer.hpp"

namespace mlengine::parametric {

class Sequential : public core::Layer {
 public:
  void add(std::shared_ptr<core::Layer> layer) { layers_.push_back(layer); }

  void forward(const core::MatrixRM& input, core::MatrixRM& output) override {
    core::MatrixRM current_input = input;
    core::MatrixRM current_output;

    for (auto& layer : layers_) {
      layer->forward(current_input, current_output);
      current_input = current_output;
    }
    output = current_input;
  }

  core::MatrixRM backward(const core::MatrixRM& output_gradient) override {
    core::MatrixRM gradient = output_gradient;
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
      gradient = (*it)->backward(gradient);
    }
    return gradient;
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