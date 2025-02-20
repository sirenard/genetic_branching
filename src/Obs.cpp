//
// Created by simon on 20/02/25.
//

#include "Obs.h"

Obs::Obs(int size): features(size), computed(size, false) {}

double Obs::operator[](int index) {
    if (!computed[index]) {
        compute(index);
    }

    return features[index];
}

void Obs::reset() {
    std::fill(computed.begin(), computed.end(), false);
}


