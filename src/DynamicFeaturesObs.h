//
// Created by simon on 14/02/25.
//

#ifndef DYNAMICFEATURESOBS_H
#define DYNAMICFEATURESOBS_H

#include <vector>

#include <scip/scip.h>

#include "Obs.h"

class DynamicFeaturesObs: public Obs{
    double nSbUp {0};
    double nSbDown {0};

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

    /**
     *
     * @return
     */
    std::vector<double> computeInfeasibilityStats();

    void compute(int index) override;
public:
    static const int size = 17;
    explicit DynamicFeaturesObs(long scipl);

};


#endif //DYNAMICFEATURESOBS_H
