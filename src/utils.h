//
// Created by simon on 21/02/25.
//

#ifndef UTILS_H
#define UTILS_H
#include <scip/scip.h>

template<typename T>
T safe_div(T a, T b);

SCIP_Var* getVarFromProbIndex(SCIP* scip, int probIndex);

#endif //UTILS_H
