#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "parametric/LogisticNeuron.hpp"

namespace py = pybind11;
using namespace mlengine::parametric;

PYBIND11_MODULE(nn_core, m) {
  m.doc() = "C++ Parametric Optimization Layer Engine for NNEngine";

  py::class_<LogisticNeuron>(m, "LogisticNeuron")
      .def(py::init<>())
      .def("fit", &LogisticNeuron::fit, py::arg("X"), py::arg("y"),
           py::arg("epochs"), py::arg("learning_rate"))
      .def("predict_proba", &LogisticNeuron::predict_proba)
      .def("predict", &LogisticNeuron::predict)
      .def("get_weights", &LogisticNeuron::get_weights)
      .def("get_bias", &LogisticNeuron::get_bias);
}