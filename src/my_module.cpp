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
        .def("computeObjCoefficient", &StaticFeaturesObs::computeObjCoefficient, "Get info 1")
        .def("computeConstraintsDegreeStatistics", &StaticFeaturesObs::computeConstraintsDegreeStatistics, "Get info 1")
        .def("computeNonZeroCoefficientsStatistics", &StaticFeaturesObs::computeNonZeroCoefficientsStatistics, "Get info 1");

    py::class_<TreeFeaturesObs>(m, "TreeFeaturesObs")
        .def(py::init<long>())
        .def("computeFeatures", &TreeFeaturesObs::computeFeatures, "Get info 1");

    py::class_<DynamicFeaturesObs>(m, "DynamicFeaturesObs")
        .def(py::init<long, int>())
        .def("getPseudoCosts", &DynamicFeaturesObs::getPseudoCosts, "Get info 1")
        .def("getInfeasibilityStatistics", &DynamicFeaturesObs::getInfeasibilityStatistics, "Get info 1")
        .def("getStrongBranchingScore", &DynamicFeaturesObs::getStrongBranchingScore, "Get info 1");
}
