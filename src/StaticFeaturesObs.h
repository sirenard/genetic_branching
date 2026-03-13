#ifndef STATICFEATURESOBS_H
#define STATICFEATURESOBS_H
#include "config.h"
#include <scip/scip.h>
#include "Obs.h"
#include "statistics.h"
#include <array>

#ifdef USE_PYTHON
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
#endif

class StaticFeaturesObs : public Obs {
private:
    std::array<double, 1> computeObjCoefficient();
    std::array<double, 9> computeNonZeroCoefficientsStatistics();
    std::array<double, 4> computeConstraintsDegreeStatistics();

public:
    static const int size = 14;
    
    explicit StaticFeaturesObs(SCIP *scip);
    #ifdef USE_PYTHON
    explicit StaticFeaturesObs(py::object py_scip);
    #endif

    
    void compute(int index) override;
};

#endif // STATICFEATURESOBS_H