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

DynamicFeaturesObs::DynamicFeaturesObs(SCIP* scip): Obs(scip, size) {
}

DynamicFeaturesObs::DynamicFeaturesObs(py::object py_scip) : DynamicFeaturesObs(
        static_cast<SCIP *>(PyCapsule_GetPointer(py_scip.ptr(), "scip"))
    ) {
    if (!scip) {
        throw py::error_already_set();
    }
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
    auto const solval = SCIPvarGetLPSol(var);
    SCIP_Real down_val = SCIPgetVarPseudocostVal(scip, var, -SCIPfrac(scip, solval));
    SCIP_Real up_val   = SCIPgetVarPseudocostVal(scip, var, 1.0 - SCIPfrac(scip, solval));

    // Fetch the final combined discounted score
    SCIP_Real score    = SCIPgetVarDPseudocostScore(scip, var, solval, 0.2);

    return {
        std::min(SCIPfeasFrac(scip, solval), 1.0 - SCIPfeasFrac(scip, solval)),
        1.0 - SCIPfeasFrac(scip, solval),
        up_val,
        down_val,
        score
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
