#pragma once

#include <memory>
#include <vector>

#include "autograd/Tensor.hpp"
#include "core/Layer.hpp"

namespace mlengine::core {

class Module : public Layer {
 protected:
  std::vector<std::shared_ptr<Layer>> modules_;

 public:
  virtual ~Module() = default;

  std::shared_ptr<Layer> add_module(std::shared_ptr<Layer> layer) {
    modules_.push_back(layer);
    return layer;
  }

  std::vector<autograd::Tensor*> parameters() override {
    std::vector<autograd::Tensor*> params;
    params.reserve(modules_.size() * 2);

    for (const auto& m : modules_) {
      auto m_params = m->parameters();
      params.insert(params.end(), m_params.begin(), m_params.end());
    }
    return params;
  }

  MatrixRM predict(Eigen::Ref<const MatrixRM> X) {
    autograd::Tape tape(false);
    autograd::Tensor* X_tensor = tape.push_tensor(X, false);
    autograd::Tensor* predictions = this->forward(&tape, X_tensor);
    return predictions->data;
  }
};

}  // namespace mlengine::core