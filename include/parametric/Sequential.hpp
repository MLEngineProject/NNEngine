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

  std::vector<autograd::Tensor*> parameters() override {
    std::vector<autograd::Tensor*> params;
    params.reserve(layers_.size() * 2);
    for (const auto& layer : layers_) {
      auto l_params = layer->parameters();
      params.insert(params.end(), l_params.begin(), l_params.end());
    }
    return params;
  }

 private:
  std::vector<std::shared_ptr<core::Layer>> layers_;
};

}  // namespace mlengine::parametric