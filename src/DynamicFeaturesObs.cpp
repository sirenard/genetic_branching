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

DynamicFeaturesObs::DynamicFeaturesObs(long scipl, int probIndex): Obs(size), scip((SCIP *) scipl) {

    int nVars = SCIPgetNVars(scip);
    auto vars = SCIPgetVars(scip);
    for (int i = 0; i < nVars; ++i) {
        auto var = vars[i];
        if (SCIPvarGetProbindex(var) == probIndex) {
            this->var = var;
        }
    }
}

void DynamicFeaturesObs::compute(int index){
    std::vector<double> tmp;
    int start = 0;
    if (index < 3) {
        tmp = getPseudoCosts();
    } else if (index < 7) {
        start = 3;
        tmp = getInfeasibilityStatistics();
    } else if (index < 9) {
        start = 7;
        tmp = getStrongBranchingScore();
    } else if (index < 11) {
        start = 9;
        tmp = getNSb();
    }

    for (int i=0; i<tmp.size(); i++) {
        features[i+start] = tmp[i];
        computed[i+start] = true;
    }
    // std::copy_backward(tmp.begin(), tmp.end(), features.begin()+start);
    // std::fill(computed.begin() + start, computed.begin() + start + tmp.size(), true);
}

std::vector<double> DynamicFeaturesObs::getPseudoCosts() {
    double value = SCIPvarGetLPSol(var);
    double downFrac = value - floor(value);
    double upFrac = 1 - downFrac;

    return {
        SCIPgetVarPseudocost(scip, var, SCIP_BRANCHDIR_UPWARDS) * upFrac,
        SCIPgetVarPseudocost(scip, var, SCIP_BRANCHDIR_DOWNWARDS) * downFrac,
        SCIPgetVarRedcost(scip, var),
    };
}

std::vector<double> DynamicFeaturesObs::getInfeasibilityStatistics() {
    auto const n_infeasibles_up = SCIPvarGetCutoffSum(var, SCIP_BRANCHDIR_UPWARDS);
    auto const n_infeasibles_down = SCIPvarGetCutoffSum(var, SCIP_BRANCHDIR_DOWNWARDS);
    auto const n_branchings_up = static_cast<double>(SCIPvarGetNBranchings(var, SCIP_BRANCHDIR_UPWARDS));
    auto const n_branchings_down = static_cast<double>(SCIPvarGetNBranchings(var, SCIP_BRANCHDIR_DOWNWARDS));

    return{
        n_infeasibles_up,
        n_infeasibles_down,
        n_branchings_up,
        n_branchings_down,
    };
}

std::vector<double> DynamicFeaturesObs::getStrongBranchingScore() {
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

    auto lp = SCIPgetLPObjval(scip);

    if (!upinf) {
        ++nSbUp;
    }

    if (!downinf) {
        ++nSbDown;
    }

    return {
        down - lp,
        up - lp,
    };
}

std::vector<double> DynamicFeaturesObs::getNSb() {
    return {
        static_cast<double>(nSbUp),
        static_cast<double>(nSbDown),
    };
}
