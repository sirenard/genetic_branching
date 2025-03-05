//
// Created by simon on 21/02/25.
//

#include "utils.h"

#include <iostream>
#include <memory>
#include <vector>

template<typename T>
T safe_div(T a, T b) {
    if (b == 0) return 0;
    return a / b;
}

template double safe_div<double>(double a, double b);

SCIP_Var * getVarFromProbIndex(SCIP* scip, int probIndex) {
    // int nCols = SCIPgetNLPCols(scip);
    // auto cols = SCIPgetLPCols(scip);

    int n = SCIPgetNLPBranchCands(scip);
    SCIP_Var** vars;
    SCIPgetLPBranchCands(scip, &vars, nullptr, nullptr, nullptr, nullptr, nullptr);

    for (int i = 0; i < n; i++) {
        auto var = vars[i];
        if (SCIPvarGetProbindex(var) == probIndex) {
            return var;
        }
    }
    assert(false);
    return nullptr;
}
