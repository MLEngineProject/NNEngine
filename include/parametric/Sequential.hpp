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
    core::MatrixRM temp = input;
    for (auto& layer : layers_) {
      layer->forward(temp, temp);
    }
    output = temp;
  }

  core::MatrixRM backward(const core::MatrixRM& output_gradient,
                          double learning_rate) override {
    core::MatrixRM gradient = output_gradient;
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
      gradient = (*it)->backward(gradient, learning_rate);
    }
    return gradient;
  }

 private:
  std::vector<std::shared_ptr<core::Layer>> layers_;
};

}  // namespace mlengine::parametric