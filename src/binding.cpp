#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

#include "core/Layer.hpp"
#include "parametric/DenseLayer.hpp"
#include "parametric/LogisticNeuron.hpp"
#include "parametric/ReLULayer.hpp"
#include "parametric/Sequential.hpp"

namespace py = pybind11;
using namespace mlengine::core;
using namespace mlengine::parametric;

PYBIND11_MODULE(nn_core, m) {
  m.doc() = "C++ Parametric Optimization Layer Engine for NNEngine";

  py::class_<Layer, std::shared_ptr<Layer>>(m, "Layer");

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
      .def("backward", &ReLULayer::backward);

  py::class_<Sequential, Layer, std::shared_ptr<Sequential>>(m, "Sequential")
      .def(py::init<>())
      .def("add", &Sequential::add, py::arg("layer"))
      .def("forward",
           [](Sequential& self, const MatrixRM& input) {
             MatrixRM output;
             self.forward(input, output);
             return output;
           })
      .def("backward", &Sequential::backward);
}