#pragma once
#include <Eigen/Core>
#include <vector>

#include "autograd/Tape.hpp"

namespace mlengine::core {

using MatrixRM = autograd::MatrixRM;

class Layer {
 public:
  virtual ~Layer() = default;

  virtual autograd::Tensor* forward(autograd::Tape& tape,
                                    autograd::Tensor* input) = 0;

  virtual std::vector<autograd::Tensor*> parameters() { return {}; }
};

}  // namespace mlengine::core