//
// Created by simon on 14/02/25.
//

#include "DynamicFeaturesObs.h"

DynamicFeaturesObs::DynamicFeaturesObs(long scipl, int probIndex): scip((SCIP*)scipl) {
    int nVars = SCIPgetNVars(scip);
    auto vars = SCIPgetVars(scip);
    for (int i=0; i<nVars; ++i) {
        auto var = vars[i];
        if (SCIPvarGetProbindex(var) == probIndex) {
            this->var = var;
        }
    }
}

std::vector<float> DynamicFeaturesObs::getPseudoCosts() {
    float value = SCIPvarGetLPSol(var);
    float downFrac = value - floor(value);
    float upFrac = 1 - downFrac;

    return {
        static_cast<float>(SCIPgetVarPseudocost(scip, var, SCIP_BRANCHDIR_UPWARDS)) / upFrac,
        static_cast<float>(SCIPgetVarPseudocost(scip, var, SCIP_BRANCHDIR_DOWNWARDS)) / downFrac,
	    static_cast<float>(SCIPgetVarRedcost(scip, var)),
    };
}

DynamicFeaturesObs::~DynamicFeaturesObs() {
    // SCIPreleaseVar(scip, &var);
}
