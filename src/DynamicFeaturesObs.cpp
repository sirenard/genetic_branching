//
// Created by simon on 14/02/25.
//

#include "DynamicFeaturesObs.h"
#include "scip/branch.h"

#include <iostream>

#include "utils.h"

bool DynamicFeaturesObs::isRowActive(SCIP_ROW *row) const {
    auto const activity = SCIPgetRowActivity(scip, row);
    auto const lhs = SCIProwGetLhs(row);
    auto const rhs = SCIProwGetRhs(row);
    return SCIProwIsInLP(row) && (SCIPisEQ(scip, activity, rhs) || SCIPisEQ(scip, activity, lhs));
}

DynamicFeaturesObs::DynamicFeaturesObs(long scipl): Obs(scipl, size) {
}

void DynamicFeaturesObs::compute(int index) {
    std::vector<double> tmp;
    int start = 0;
    if (index < 5) {
        tmp = getPseudoCosts();
    } else if (index < 9) {
        start = 5;
        tmp = getInfeasibilityStatistics();
    } else if (index < 12) {
        start = 9;
        tmp = getStrongBranchingScore();
    } else if (index < 14) {
        start = 12;
        tmp = getNSb();
    }

    for (int i = 0; i < tmp.size(); i++) {
        features[i + start] = tmp[i];
        computed[i + start] = true;
    }
}

std::vector<double> DynamicFeaturesObs::getPseudoCosts() {
    auto col = SCIPvarGetCol(var);

    auto const solval = SCIPcolGetPrimsol(col);
    auto const floor_distance = SCIPfeasFrac(scip, solval);
    auto const ceil_distance = 1. - floor_distance;
    auto const weighted_pseudocost_up = ceil_distance * SCIPgetVarPseudocost(scip, var, SCIP_BRANCHDIR_UPWARDS);
    auto const weighted_pseudocost_down = floor_distance * SCIPgetVarPseudocost(scip, var, SCIP_BRANCHDIR_DOWNWARDS);
    auto constexpr epsilon = 1e-5;
    auto const wpu_approx = std::max(weighted_pseudocost_up, epsilon);
    auto const wpd_approx = std::max(weighted_pseudocost_down, epsilon);
    auto const weighted_pseudocost_ratio = safe_div<double>(std::min(wpu_approx, wpd_approx),
                                                            std::max(wpu_approx, wpd_approx));
    double gain[2] = {weighted_pseudocost_down, weighted_pseudocost_up};

    return {
        std::min(floor_distance, ceil_distance),
        ceil_distance,
        weighted_pseudocost_up,
        weighted_pseudocost_down,
        // SCIPgetVarPseudocost(scip, var, SCIP_BRANCHDIR_UPWARDS),
        // SCIPgetVarPseudocost(scip, var, SCIP_BRANCHDIR_DOWNWARDS),
        //weighted_pseudocost_ratio,
        SCIPgetBranchScoreMultiple(scip, var, 2, gain),
    };
}

std::vector<double> DynamicFeaturesObs::getInfeasibilityStatistics() {
    auto const n_infeasibles_up = SCIPvarGetCutoffSum(var, SCIP_BRANCHDIR_UPWARDS);
    auto const n_infeasibles_down = SCIPvarGetCutoffSum(var, SCIP_BRANCHDIR_DOWNWARDS);
    auto const n_branchings_up = static_cast<double>(SCIPvarGetNBranchings(var, SCIP_BRANCHDIR_UPWARDS));
    auto const n_branchings_down = static_cast<double>(SCIPvarGetNBranchings(var, SCIP_BRANCHDIR_DOWNWARDS));
    return {
        n_infeasibles_up,
        n_infeasibles_down,
        n_branchings_up,
        n_branchings_down,
    };
}

std::vector<double> DynamicFeaturesObs::getStrongBranchingScore() {
    if ( SCIPgetNLPBranchCands(scip) == 1 ) {
        auto tmp = getPseudoCosts();
        return {
            tmp[2],
            tmp[3],
            tmp[4],
        };
    }


    int itlim = INT_MAX;
    double up = -SCIPinfinity(scip);
    double down = -SCIPinfinity(scip);
    unsigned int downvalid;
    unsigned int upvalid;
    unsigned int downinf;
    unsigned int upinf;
    unsigned int downconflict;
    unsigned int upconflict;
    unsigned int lperror;
    auto lpobjval = SCIPgetLPObjval(scip);
    auto val = SCIPvarGetLPSol(var);

    SCIPstartStrongbranch(scip, 0);

    SCIPgetVarStrongbranchFrac(
        scip,
        var,
        itlim,
        0,
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

    down = MAX(down, lpobjval);
    up = MAX(up, lpobjval);
    double downgain = down - lpobjval;
    double upgain = up - lpobjval;

    /* update variable pseudo cost values */
    if ( !downinf && downvalid ) {
        SCIPupdateVarPseudocost(scip, var, 0.0 - SCIPfrac(scip, val), downgain, 1.0);
    }
    if ( !upinf && upvalid ) {
        SCIPupdateVarPseudocost(scip, var, 1.0 - SCIPfrac(scip, val), upgain, 1.0);
    }

    SCIPendStrongbranch(scip);

    double gain[2] = {downgain, upgain};

    return {
        downgain,
        upgain,
        SCIPgetBranchScoreMultiple(scip, var, 2, gain),
    };
}

std::vector<double> DynamicFeaturesObs::getNSb() {
    return {
        SCIPgetVarPseudocostCountCurrentRun(scip, var, SCIP_BRANCHDIR_UPWARDS),
        SCIPgetVarPseudocostCountCurrentRun(scip, var, SCIP_BRANCHDIR_DOWNWARDS),
    };
}
