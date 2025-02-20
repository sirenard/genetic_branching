//
// Created by simon on 14/02/25.
//

#ifndef STATICFEATURESOBS_H
#define STATICFEATURESOBS_H

#include <iostream>
#include <vector>

#include <scip/scip.h>

#include "ArrayView.h"
#include "Obs.h"
#include "statistics.h"

class StaticFeaturesObs: public Obs{
    SCIP* scip;
    SCIP_VAR* var {};

    /**
     * Get the variable objective's coefficient
     * @return
     */
    std::vector<double> computeObjCoefficient(){

        return {static_cast<double>(SCIPvarGetObj(var))};
    }

    /**
     * Statistics about variable's constraint coefficient. Report the number of constraint in which it is active
     * And statistics on positive/negative coefficient (mean, stdev, min, max)
     * @return vector of 9 features
     */
    std::vector<double> computeNonZeroCoefficientsStatistics(){
        auto col = SCIPvarGetCol(var);

        int count = SCIPcolGetNLPNonz(col);
        auto data = ArrayView(SCIPcolGetVals(col), count);

        auto positiveStats = statistics<ArrayView<double>, double>(data, [](double val){return val > 0;;});
        auto negativeStats = statistics<ArrayView<double>, double>(data, [](double val){return val < 0;});

        return {
            static_cast<double>(count),
            positiveStats.mean,
            positiveStats.stdev,
            positiveStats.min,
            positiveStats.max,
            negativeStats.mean,
            negativeStats.stdev,
            negativeStats.min,
            negativeStats.max,
        };
    }

    /**
    * Stats. for constraint degrees.
    *
    * The degree of a constraint is the number of variables that participate in it.
    * A variable may participate in multiple constraints, and statistics over those constraints'
    * degrees are used. (mean, stdev, min, max)
    * @return Vector of 4 features
    */
    std::vector<double> computeConstraintsDegreeStatistics() {
        std::vector<double> degrees;

        auto col = SCIPvarGetCol(var);
        auto const n_rows = SCIPcolGetNNonz(col);
        auto rows = SCIPcolGetRows(col);

        for (int i=0; i<n_rows; i++) {
            auto row = rows[i];
            degrees.push_back(SCIProwGetNNonz(row));
        }

        auto stats = statistics<std::vector<double>, double>(degrees);


        return {
            stats.mean,
            stats.stdev,
            stats.min,
            stats.max,
        };
    }

    void compute(int index) override {
        std::vector<double> tmp;
        int start = 0;
        if (index < 1) {
            tmp = computeObjCoefficient();
        } else if (index < 10) {
            start = 1;
            tmp = computeNonZeroCoefficientsStatistics();
        } else if (index < 14) {
            start = 10;
            tmp = computeConstraintsDegreeStatistics();
        }

        for (int i=0; i<tmp.size(); i++) {
            features[i+start] = tmp[i];
            computed[i+start] = true;
        }
        // std::copy_backward(tmp.begin(), tmp.end(), features.begin()+start);
        // std::fill(computed.begin() + start, computed.begin() + start + tmp.size(), true);
    }
public:
    static const int size = 14;
    StaticFeaturesObs(long scipl, int probIndex): Obs(14), scip((SCIP*)scipl) {
        int nVars = SCIPgetNVars(scip);
        auto vars = SCIPgetVars(scip);
        for (int i=0; i<nVars; ++i) {
            auto var = vars[i];
            // SCIP_VAR* transformedVar;
            // SCIPgetTransformedVar(scip, var, &transformedVar);
            if (SCIPvarGetProbindex(var) == probIndex) {
                this->var = var;
            }
        }

    }


};



#endif //STATICFEATURESOBS_H
