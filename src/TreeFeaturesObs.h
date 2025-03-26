//
// Created by simon on 14/02/25.
//

#ifndef TREEFEATURESOBS_H
#define TREEFEATURESOBS_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include "Obs.h"

class TreeFeaturesObs: public Obs {
    double gap();
    double leafFrequency();
    double openNodes();
    double ssg();
    double treeWeight();
    double completion();
    double depth();


public:
    static const int size = 5;
    explicit TreeFeaturesObs(SCIP* scip);
    explicit TreeFeaturesObs(py::object pyscip);

    void compute(int index) override;
};



#endif //TREEFEATURESOBS_H
