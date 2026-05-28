#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

#include "core/Layer.hpp"
#include "core/Loss.hpp"
#include "core/Model.hpp"
#include "core/Optimizer.hpp"
#include "parametric/DenseLayer.hpp"
#include "parametric/LeakyReLULayer.hpp"
#include "parametric/ReLULayer.hpp"
#include "parametric/Sequential.hpp"
#include "parametric/SoftmaxLayer.hpp"

namespace py = pybind11;
using namespace mlengine::core;
using namespace mlengine::parametric;

PYBIND11_MODULE(nn_core, m) {
  m.doc() = "C++ Parametric Optimization Layer Engine for NNEngine";

  py::class_<Layer, std::shared_ptr<Layer>>(m, "Layer");
  py::class_<Loss, std::shared_ptr<Loss>>(m, "Loss");
  py::class_<Optimizer, std::shared_ptr<Optimizer>>(m, "Optimizer");

  py::class_<MSELoss, Loss, std::shared_ptr<MSELoss>>(m, "MSELoss")
      .def(py::init<>());
  py::class_<CategoricalCrossEntropyLoss, Loss,
             std::shared_ptr<CategoricalCrossEntropyLoss>>(
      m, "CategoricalCrossEntropyLoss")
      .def(py::init<>());

  py::class_<SGD, Optimizer, std::shared_ptr<SGD>>(m, "SGD").def(
      py::init<double>(), py::arg("learning_rate") = 0.01);

  py::class_<Adam, Optimizer, std::shared_ptr<Adam>>(m, "Adam").def(
      py::init<double>(), py::arg("learning_rate") = 0.001);

  py::class_<DenseLayer, Layer, std::shared_ptr<DenseLayer>>(m, "DenseLayer")
      .def(py::init<int, int>())
      .def("get_weights", &DenseLayer::get_weights)
      .def("get_bias", &DenseLayer::get_bias);

  py::class_<ReLULayer, Layer, std::shared_ptr<ReLULayer>>(m, "ReLULayer")
      .def(py::init<>());

  py::class_<LeakyReLULayer, Layer, std::shared_ptr<LeakyReLULayer>>(
      m, "LeakyReLULayer")
      .def(py::init<double>(), py::arg("alpha") = 0.01);

  py::class_<SoftmaxLayer, Layer, std::shared_ptr<SoftmaxLayer>>(m,
                                                                 "SoftmaxLayer")
      .def(py::init<>());

  py::class_<Sequential, Layer, std::shared_ptr<Sequential>>(m, "Sequential")
      .def(py::init<>())
      .def("add", &Sequential::add, py::arg("layer"));

  py::class_<Model, std::shared_ptr<Model>>(m, "Model")
      .def(py::init<>())
      .def("add", &Model::add, py::arg("layer"))
      .def("compile", &Model::compile, py::arg("optimizer"), py::arg("loss_fn"))
      .def("fit", &Model::fit, py::arg("X"), py::arg("y"),
           py::arg("epochs") = 100, py::arg("batch_size") = 32,
           py::arg("tol") = 1e-4, py::arg("n_iter_no_change") = 10,
           py::arg("verbose") = true)
      .def("predict", &Model::predict, py::arg("X"));
}