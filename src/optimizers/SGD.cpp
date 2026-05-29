#include "optimizers/SGD.hpp"

namespace mlengine::core {

SGD::SGD(float learning_rate) : Optimizer(learning_rate) {}

void SGD::step() {
  for (auto* p : parameters_) {
    if (p->requires_grad) {
      p->data -= lr_ * p->grad;
    }
  }
}

}  // namespace mlengine::core
