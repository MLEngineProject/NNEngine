#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

#include "autograd/Tape.hpp"
#include "autograd/Tensor.hpp"
#include "core/DataLoader.hpp"
#include "core/JITGraph.hpp"
#include "core/Layer.hpp"
#include "core/Loss.hpp"
#include "core/Module.hpp"
#include "core/Optimizer.hpp"
#include "core/Random.hpp"
#include "core/Regularizer.hpp"
#include "layers/DenseLayer.hpp"
#include "layers/LeakyReLULayer.hpp"
#include "layers/ReLULayer.hpp"
#include "losses/MSELoss.hpp"
#include "losses/SoftmaxCrossEntropyLoss.hpp"
#include "optimizers/Adam.hpp"
#include "optimizers/SGD.hpp"
#include "regularizers/L2Regularizer.hpp"

namespace py = pybind11;
using namespace mlengine::core;
using namespace mlengine::layers;

class PyModule : public Module {
 public:
  using Module::Module;

  mlengine::autograd::Tensor* forward(
      mlengine::autograd::Tape* tape,
      mlengine::autograd::Tensor* input) override {
    PYBIND11_OVERRIDE_PURE(mlengine::autograd::Tensor*, Module, forward, tape,
                           input);
  }
};

void bind_core_utils(py::module_& m) {
  m.def("set_seed", &set_seed, py::arg("seed"), "Seed RNG.");

  py::class_<mlengine::autograd::Tensor>(m, "Tensor")
      .def_readonly("data", &mlengine::autograd::Tensor::data)
      .def_readonly("grad", &mlengine::autograd::Tensor::grad)
      .def_readonly("requires_grad",
                    &mlengine::autograd::Tensor::requires_grad);

  py::class_<mlengine::autograd::Tape,
             std::shared_ptr<mlengine::autograd::Tape>>(m, "Tape")
      .def(py::init<bool>(), py::arg("record_ops") = true)
      .def("push_tensor", &mlengine::autograd::Tape::push_tensor,
           py::arg("data"), py::arg("requires_grad") = true,
           py::return_value_policy::reference)
      .def("add", &mlengine::autograd::Tape::add_bias,
           py::return_value_policy::reference)
      .def("backward", &mlengine::autograd::Tape::backward)
      .def("reset", &mlengine::autograd::Tape::reset);

  py::class_<Layer, std::shared_ptr<Layer>>(m, "Layer")
      .def("parameters", &Layer::parameters);

  py::class_<Module, Layer, PyModule, std::shared_ptr<Module>>(m, "Module")
      .def(py::init<>())
      .def("forward", &Module::forward, py::return_value_policy::reference)
      .def("predict", &Module::predict,
           py::call_guard<py::gil_scoped_release>())
      .def("parameters", &Module::parameters)
      .def("add_module", &Module::add_module, py::arg("layer"),
           py::return_value_policy::reference);

  py::class_<DataLoader>(m, "DataLoader")
      .def(py::init<const mlengine::MatrixRM&, const mlengine::MatrixRM&,
                    size_t, bool, bool>(),
           py::arg("X"), py::arg("y"), py::arg("batch_size"),
           py::arg("shuffle") = true, py::arg("drop_last") = false)
      .def("reset", &DataLoader::reset);
}

void bind_losses_and_regs(py::module_& m) {
  py::class_<Loss, std::shared_ptr<Loss>>(m, "Loss").def(
      "forward", &Loss::forward, py::arg("predictions"), py::arg("targets"));
  py::class_<MSELoss, Loss, std::shared_ptr<MSELoss>>(m, "MSELoss")
      .def(py::init<>());
  py::class_<SoftmaxCrossEntropyLoss, Loss,
             std::shared_ptr<SoftmaxCrossEntropyLoss>>(
      m, "SoftmaxCrossEntropyLoss")
      .def(py::init<>());
  py::class_<Regularizer, std::shared_ptr<Regularizer>>(m, "Regularizer");
  py::class_<L2Regularizer, Regularizer, std::shared_ptr<L2Regularizer>>(
      m, "L2Regularizer")
      .def(py::init<float>(), py::arg("l2") = 0.0001f);
}

void bind_optimizers(py::module_& m) {
  py::class_<Optimizer, std::shared_ptr<Optimizer>>(m, "Optimizer");
  py::class_<SGD, Optimizer, std::shared_ptr<SGD>>(m, "SGD").def(
      py::init<float>(), py::arg("learning_rate") = 0.01f);
  py::class_<Adam, Optimizer, std::shared_ptr<Adam>>(m, "Adam").def(
      py::init<float>(), py::arg("learning_rate") = 0.001f);
}

void bind_layers(py::module_& m) {
  py::class_<DenseLayer, Layer, std::shared_ptr<DenseLayer>>(m, "DenseLayer")
      .def(py::init<int, int>())
      .def("forward", &DenseLayer::forward, py::return_value_policy::reference)
      .def("get_weights", &DenseLayer::get_weights)
      .def("get_bias", &DenseLayer::get_bias);

  py::class_<ReLULayer, Layer, std::shared_ptr<ReLULayer>>(m, "ReLULayer")
      .def(py::init<>())
      .def("forward", &ReLULayer::forward, py::return_value_policy::reference);

  py::class_<LeakyReLULayer, Layer, std::shared_ptr<LeakyReLULayer>>(
      m, "LeakyReLULayer")
      .def(py::init<float>(), py::arg("alpha") = 0.01f)
      .def("forward", &LeakyReLULayer::forward,
           py::return_value_policy::reference);
}

void bind_model(py::module_& m) {
  py::class_<JITGraph, std::shared_ptr<JITGraph>>(m, "JITGraph")
      .def(py::init<std::shared_ptr<Layer>, std::shared_ptr<Optimizer>,
                    std::shared_ptr<Loss>, std::shared_ptr<Regularizer>>(),
           py::arg("model"), py::arg("optimizer"), py::arg("loss_fn"),
           py::arg("regularizer") = nullptr)
      .def("trace_batch", &JITGraph::trace_batch, py::arg("dataloader"))
      .def("fast_loop", &JITGraph::fast_loop, py::arg("dataloader"),
           py::call_guard<py::gil_scoped_release>())
      .def("fast_fit", &JITGraph::fast_fit, py::arg("dataloader"),
           py::arg("epochs"), py::arg("tol") = 1e-4f,
           py::arg("n_iter_no_change") = 10, py::arg("verbose") = true,
           py::call_guard<py::gil_scoped_release>());
}

PYBIND11_MODULE(nn_core, m) {
  m.doc() = "C++ Core Engine for NNEngine";
  bind_core_utils(m);
  bind_losses_and_regs(m);
  bind_optimizers(m);
  bind_layers(m);
  bind_model(m);
}