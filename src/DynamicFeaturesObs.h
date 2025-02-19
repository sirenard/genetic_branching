//
// Created by simon on 14/02/25.
//

#ifndef DYNAMICFEATURESOBS_H
#define DYNAMICFEATURESOBS_H

#include <vector>

#include <scip/scip.h>

class DynamicFeaturesObs{
    SCIP* scip;
    SCIP_VAR* var {};

    bool isRowActive(SCIP_ROW* row) const;
public:
    DynamicFeaturesObs(long scipl, int probIndex);

    /**
     * Compute weighted pseudocost (downward and upward) and variable's reduced cost
     * @return Vector of 3 features
     */
    std::vector<float> getPseudoCosts();

    /**
     * Infeasibility statistics.
     * Number and fraction of nodes for which applying SB to variable xj led to one (two) infeasible children (during data collection).
     * @return Vector of 4 features
     */
    std::vector<float> getInfeasibilityStatistics();

    std::vector<float> getStrongBranchingScore();

};


#endif //DYNAMICFEATURESOBS_H
