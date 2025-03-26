//
// Created by simon on 20/02/25.
//

#include "Obs.h"

#include <iostream>

#include "utils.h"

Obs::Obs(SCIP* scip, int size): features(size), computed(size, false), scip(scip) {}

double Obs::operator[](int index) {
    if (!computed[index]) {
        compute(index);
    }

    return features[index];
}

void Obs::setVar(int probIndex) {
    if (probIndex >= 0) {
        var = getVarFromProbIndex(scip, probIndex);
    }
}

void Obs::reset() {
    std::fill(computed.begin(), computed.end(), false);
}


