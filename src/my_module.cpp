#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <scip/scip.h>
#include "StaticFeaturesObs.h"
#include "TreeFeaturesObs.h"
#include "DynamicFeaturesObs.h"

namespace py = pybind11;

void tmp(long scipl) {
    SCIP* scip = (SCIP*) scipl;
    std::cout << SCIPgetStage(scip) << std::endl;;
}

PYBIND11_MODULE(my_module, m) {
    m.doc() = "pybind11 example plugin";

    py::class_<StaticFeaturesObs>(m, "StaticFeaturesObs")
        .def(py::init<long, int>())
        .def("__getitem__", &StaticFeaturesObs::operator[], "Get feature at index i")
        .def("__len__", [](){return StaticFeaturesObs::size;}, "Get size")
        .def("reset", &StaticFeaturesObs::reset, "Reset values")
        .def_static("size", [](){return StaticFeaturesObs::size;}, "Get size");

    py::class_<TreeFeaturesObs>(m, "TreeFeaturesObs")
        .def(py::init<long>())
        .def("__getitem__", &TreeFeaturesObs::operator[], "Get feature at index i")
        .def("__len__", [](){return TreeFeaturesObs::size;}, "Get size")
        .def("reset", &TreeFeaturesObs::reset, "Reset values")
        .def_static("size", [](){return TreeFeaturesObs::size;}, "Get size");

    py::class_<DynamicFeaturesObs>(m, "DynamicFeaturesObs")
        .def(py::init<long, int>())
        .def("__getitem__", &DynamicFeaturesObs::operator[], "Get feature at index i")
        .def("__len__", [](){return DynamicFeaturesObs::size;}, "Get size")
        .def("reset", &DynamicFeaturesObs::reset, "Reset values")
        .def_static("size", [](){return DynamicFeaturesObs::size;}, "Get size");
}
