#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

#include "core/Layer.hpp"
#include "core/Loss.hpp"
#include "core/Model.hpp"
#include "parametric/DenseLayer.hpp"
#include "parametric/LogisticNeuron.hpp"
#include "parametric/ReLULayer.hpp"
#include "parametric/Sequential.hpp"
#include "parametric/SoftmaxLayer.hpp"

namespace py = pybind11;
using namespace mlengine::core;
using namespace mlengine::parametric;

PYBIND11_MODULE(nn_core, m) {
  m.doc() = "C++ Parametric Optimization Layer Engine for NNEngine";

  py::class_<Layer, std::shared_ptr<Layer>>(m, "Layer")
      .def("update_weights", &Layer::update_weights);

  py::class_<Loss, std::shared_ptr<Loss>>(m, "Loss");

  py::class_<MSELoss, Loss, std::shared_ptr<MSELoss>>(m, "MSELoss")
      .def(py::init<>())
      .def("calculate", &MSELoss::calculate)
      .def("backward", &MSELoss::backward);

  py::class_<CategoricalCrossEntropyLoss, Loss,
             std::shared_ptr<CategoricalCrossEntropyLoss>>(
      m, "CategoricalCrossEntropyLoss")
      .def(py::init<>())
      .def("calculate", &CategoricalCrossEntropyLoss::calculate)
      .def("backward", &CategoricalCrossEntropyLoss::backward);

  py::class_<LogisticNeuron>(m, "LogisticNeuron")
      .def(py::init<>())
      .def("fit", &LogisticNeuron::fit, py::arg("X"), py::arg("y"),
           py::arg("epochs"), py::arg("learning_rate"))
      .def("predict_proba", &LogisticNeuron::predict_proba)
      .def("predict", &LogisticNeuron::predict)
      .def("get_weights", &LogisticNeuron::get_weights)
      .def("get_bias", &LogisticNeuron::get_bias);

  py::class_<DenseLayer, Layer, std::shared_ptr<DenseLayer>>(m, "DenseLayer")
      .def(py::init<int, int>())
      .def("forward",
           [](DenseLayer& self, const MatrixRM& input) {
             MatrixRM output;
             self.forward(input, output);
             return output;
           })
      .def("backward", &DenseLayer::backward)
      .def("update_weights", &DenseLayer::update_weights)
      .def("get_weights", &DenseLayer::get_weights)
      .def("get_bias", &DenseLayer::get_bias);

  py::class_<ReLULayer, Layer, std::shared_ptr<ReLULayer>>(m, "ReLULayer")
      .def(py::init<>())
      .def("forward",
           [](ReLULayer& self, const MatrixRM& input) {
             MatrixRM output;
             self.forward(input, output);
             return output;
           })
      .def("backward", &ReLULayer::backward)
      .def("update_weights", &ReLULayer::update_weights);

  py::class_<SoftmaxLayer, Layer, std::shared_ptr<SoftmaxLayer>>(m,
                                                                 "SoftmaxLayer")
      .def(py::init<>())
      .def("forward",
           [](SoftmaxLayer& self, const MatrixRM& input) {
             MatrixRM output;
             self.forward(input, output);
             return output;
           })
      .def("backward", &SoftmaxLayer::backward)
      .def("update_weights", &SoftmaxLayer::update_weights);

  py::class_<Sequential, Layer, std::shared_ptr<Sequential>>(m, "Sequential")
      .def(py::init<>())
      .def("add", &Sequential::add, py::arg("layer"))
      .def("forward",
           [](Sequential& self, const MatrixRM& input) {
             MatrixRM output;
             self.forward(input, output);
             return output;
           })
      .def("backward", &Sequential::backward)
      .def("update_weights", &Sequential::update_weights);

  py::class_<Model, std::shared_ptr<Model>>(m, "Model")
      .def(py::init<>())
      .def("add", &Model::add, py::arg("layer"))
      .def("compile", &Model::compile, py::arg("loss_fn"))
      .def("fit", &Model::fit, py::arg("X"), py::arg("y"),
           py::arg("epochs") = 100, py::arg("learning_rate") = 0.01,
           py::arg("batch_size") = 32, py::arg("verbose") = true)
      .def("predict", &Model::predict, py::arg("X"));
}