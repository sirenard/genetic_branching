//
// Created by simon on 14/02/25.
//

#include "DynamicFeaturesObs.h"

bool DynamicFeaturesObs::isRowActive(SCIP_ROW *row) const {
    auto const activity = SCIPgetRowActivity(scip, row);
    auto const lhs = SCIProwGetLhs(row);
    auto const rhs = SCIProwGetRhs(row);
    return SCIProwIsInLP(row) && (SCIPisEQ(scip, activity, rhs) || SCIPisEQ(scip, activity, lhs));
}

DynamicFeaturesObs::DynamicFeaturesObs(long scipl, int probIndex): scip((SCIP *) scipl) {
    int nVars = SCIPgetNVars(scip);
    auto vars = SCIPgetVars(scip);
    for (int i = 0; i < nVars; ++i) {
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
        static_cast<float>(SCIPgetVarPseudocost(scip, var, SCIP_BRANCHDIR_UPWARDS)) * upFrac,
        static_cast<float>(SCIPgetVarPseudocost(scip, var, SCIP_BRANCHDIR_DOWNWARDS)) * downFrac,
        static_cast<float>(SCIPgetVarRedcost(scip, var)),
    };
}

std::vector<float> DynamicFeaturesObs::getInfeasibilityStatistics() {
    auto const n_infeasibles_up = static_cast<float>(SCIPvarGetCutoffSum(var, SCIP_BRANCHDIR_UPWARDS));
    auto const n_infeasibles_down = static_cast<float>(SCIPvarGetCutoffSum(var, SCIP_BRANCHDIR_DOWNWARDS));
    auto const n_branchings_up = static_cast<float>(SCIPvarGetNBranchings(var, SCIP_BRANCHDIR_UPWARDS));
    auto const n_branchings_down = static_cast<float>(SCIPvarGetNBranchings(var, SCIP_BRANCHDIR_DOWNWARDS));

    return{
        n_infeasibles_up,
        n_infeasibles_down,
        n_branchings_up,
        n_branchings_down,
    };
}

std::vector<float> DynamicFeaturesObs::getStrongBranchingScore() {
   int itlim = INT32_MAX;
   double down;
   double up;
   unsigned int downvalid;
   unsigned int upvalid;
   unsigned int downinf;
   unsigned int upinf;
   unsigned int downconflict;
   unsigned int upconflict;
   unsigned int lperror;

    SCIPgetVarsStrongbranchesFrac(
        scip,
        &var,
        1,
        itlim,
        &down,
        &up,
        &downvalid,
        &upvalid,
        &downinf,
        &upinf,
        &downconflict,
        &upconflict,
        &lperror
    );

    auto lp = static_cast<float>(SCIPgetLPObjval(scip));

    return {
        static_cast<float>(down) - lp,
        static_cast<float>(up) - lp,
    };
}
