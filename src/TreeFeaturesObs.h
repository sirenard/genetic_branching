//
// Created by simon on 14/02/25.
//

#ifndef TREEFEATURESOBS_H
#define TREEFEATURESOBS_H
#include <array>

#include "config.h"
#ifdef USE_PYTHON
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
#endif

#include "Obs.h"

class TreeFeaturesObs: public Obs<TreeFeaturesObs, 5> {
    std::array<double, 1> gap();
    std::array<double, 1> leafFrequency();
    std::array<double, 1> treeWeight();
    std::array<double, 1> completion();
    std::array<double, 1> depth();


public:
    static const int size = 5;
    explicit TreeFeaturesObs(SCIP* scip);
    #ifdef USE_PYTHON
    explicit TreeFeaturesObs(py::object pyscip);
    #endif

    void compute(int index);
};



#endif //TREEFEATURESOBS_H
