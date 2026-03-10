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

SCIP_Var* getVarFromProbIndex(SCIP* scip, int probIndex) {
    SCIP_VAR** vars = SCIPgetVars(scip);
    int nvars = SCIPgetNVars(scip);

    if (probIndex >= 0 && probIndex < nvars) {
        return vars[probIndex];
    }
    return nullptr;
}
