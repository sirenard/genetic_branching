#ifndef STATICFEATURESOBS_H
#define STATICFEATURESOBS_H

#include <iostream>
#include <vector>
#include <scip/scip.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ArrayView.h"
#include "Obs.h"
#include "statistics.h"

namespace py = pybind11;

class StaticFeaturesObs : public Obs {
private:
    std::vector<double> computeObjCoefficient();
    std::vector<double> computeNonZeroCoefficientsStatistics();
    std::vector<double> computeConstraintsDegreeStatistics();

public:
    static const int size = 14;
    
    explicit StaticFeaturesObs(SCIP *scip);
    explicit StaticFeaturesObs(py::object py_scip);
    
    void compute(int index) override;
};

#endif // STATICFEATURESOBS_H