#pragma once

#include <Eigen/Core>
#include <vector>

#include "autograd/Tape.hpp"
#include "core/Types.hpp"

namespace mlengine::core {

using MatrixRM = mlengine::MatrixRM;

/**
 * @brief Abstract building block for differentiable model components.
 */
class Layer {
 public:
  virtual ~Layer() = default;

  /**
   * @brief Run the layer on tape-owned input storage.
   * @param tape Autograd tape that owns all intermediate tensors.
   * @param input Input tensor to transform.
   * @return Pointer to the tape-owned output tensor.
   * @note Implementations must allocate outputs from the tape; the returned
   *     tensor is arena-allocated and owned by the tape, not the caller.
   */
  virtual autograd::Tensor* forward(autograd::Tape* tape,
                                    autograd::Tensor* input) = 0;

  /**
   * @brief Return mutable pointers to trainable parameters.
   * @return Trainable tensors owned by the layer.
   */
  virtual std::vector<autograd::Tensor*> parameters() { return {}; }
};

}  // namespace mlengine::core