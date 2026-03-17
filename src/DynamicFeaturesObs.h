//
// Created by simon on 14/02/25.
//

#ifndef DYNAMICFEATURESOBS_H
#define DYNAMICFEATURESOBS_H

#include <vector>

#include <scip/scip.h>
#include <array>

#include "config.h"

#ifdef USE_PYTHON
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
#endif


#include "Obs.h"

class DynamicFeaturesObs: public Obs<DynamicFeaturesObs, 14>{
    bool isRowActive(SCIP_ROW* row) const;

    /**
     * Compute weighted pseudocost (downward and upward) and variable's reduced cost
     * @return Vector of 3 features
     */
    std::array<double, 5> getPseudoCosts();

    /**
     * Infeasibility statistics.
     * Number and fraction of nodes for which applying SB to variable xj led to one (two) infeasible children (during data collection).
     * @return Vector of 4 features
     */
    std::array<double, 4> getInfeasibilityStatistics();

    /**
     * Compute and return the up and down improvement on the potential child of the variable
     * @return Vector of 2 features
     */
    std::array<double, 3> getStrongBranchingScore();

    /**
     * Return how many times the up and down improvement have been computed and valid
     * @return Vector of 2 features
     */
    std::array<double, 2> getNSb();

public:
    static const int size = 14;
    explicit DynamicFeaturesObs(SCIP* scip);
    void compute(int index);

    #ifdef USE_PYTHON
    explicit DynamicFeaturesObs(py::object py_scip);
    #endif

};


#endif //DYNAMICFEATURESOBS_H
