//
// Created by simon on 14/02/25.
//

#ifndef STATICFEATURESOBS_H
#define STATICFEATURESOBS_H

#include <iostream>
#include <vector>

#include <scip/scip.h>

#include "ArrayView.h"
#include "statistics.h"

class StaticFeaturesObs{
    SCIP* scip;
    SCIP_VAR* var {};
public:
    StaticFeaturesObs(long scipl, int probIndex): scip((SCIP*)scipl) {
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

    std::vector<float> computeObjCoefficient(){

        return {static_cast<float>(SCIPvarGetObj(var))};
    }

    std::vector<float> computeNonZeroCoefficientsStatistics(){
        auto col = SCIPvarGetCol(var);

        int count = SCIPcolGetNLPNonz(col);
        auto data = ArrayView(SCIPcolGetVals(col), count);

        auto positiveStats = statistics<ArrayView<double>, float>(data, [](float val){return val > 0;;});
        auto negativeStats = statistics<ArrayView<double>, float>(data, [](float val){return val < 0;});

        return {
            static_cast<float>(count),
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

    std::vector<float> computeConstraintsDegreeStatistics() {
        std::vector<float> degrees;

        auto col = SCIPvarGetCol(var);
        auto const n_rows = SCIPcolGetNNonz(col);
        auto rows = SCIPcolGetRows(col);

        for (int i=0; i<n_rows; i++) {
            auto row = rows[i];
            degrees.push_back(SCIProwGetNNonz(row));
        }

        auto stats = statistics<std::vector<float>, float>(degrees);


        return {
            stats.mean,
            stats.stdev,
            stats.min,
            stats.max,
        };
    }

    ~StaticFeaturesObs() {
        // SCIPreleaseVar(scip, &var);
    }
};



#endif //STATICFEATURESOBS_H
