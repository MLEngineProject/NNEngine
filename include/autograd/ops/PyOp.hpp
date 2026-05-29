#pragma once

#include <pybind11/pybind11.h>

#include "autograd/Op.hpp"

namespace mlengine::autograd::ops {

/**
 * @brief Python trampoline that forwards virtual calls into Python overrides.
 *
 * Python users can inherit from Op to define custom forward/backward passes
 * while still participating in the native tape replay loop.
 */
class PyOp : public mlengine::autograd::Op {
 public:
  using mlengine::autograd::Op::Op;

  void forward() override {
    PYBIND11_OVERRIDE_PURE(void, mlengine::autograd::Op, forward);
  }

  void backward() override {
    PYBIND11_OVERRIDE_PURE(void, mlengine::autograd::Op, backward);
  }
};

}  // namespace mlengine::autograd::ops