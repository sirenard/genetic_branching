//
// Created by simon on 14/02/25.
//

#ifndef DYNAMICFEATURESOBS_H
#define DYNAMICFEATURESOBS_H

#include <vector>

#include <scip/scip.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include "Obs.h"

class DynamicFeaturesObs: public Obs{
    bool isRowActive(SCIP_ROW* row) const;

    /**
     * Compute weighted pseudocost (downward and upward) and variable's reduced cost
     * @return Vector of 3 features
     */
    std::vector<double> getPseudoCosts();

    /**
     * Infeasibility statistics.
     * Number and fraction of nodes for which applying SB to variable xj led to one (two) infeasible children (during data collection).
     * @return Vector of 4 features
     */
    std::vector<double> getInfeasibilityStatistics();

    /**
     * Compute and return the up and down improvement on the potential child of the variable
     * @return Vector of 2 features
     */
    std::vector<double> getStrongBranchingScore();

    /**
     * Return how many times the up and down improvement have been computed and valid
     * @return Vector of 2 features
     */
    std::vector<double> getNSb();

    void compute(int index) override;
public:
    static const int size = 14;
    explicit DynamicFeaturesObs(SCIP* scip);
    explicit DynamicFeaturesObs(py::object py_scip);

};


#endif //DYNAMICFEATURESOBS_H
