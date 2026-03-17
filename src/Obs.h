#ifndef OBS_H
#define OBS_H

#include <array>
#include <scip/scip.h>

#include "utils.h"

template <typename Derived, size_t Size>
class Obs {
protected:
    std::array<double, Size> features{};
    std::array<bool, Size> computed{}; 
    SCIP* scip;
    SCIP_Var* var {};

    template<size_t S>
    void assign_features(const std::array<double, S>& tmp, int start) {
        for (size_t i = 0; i < S; i++) {
            features[i + start] = tmp[i];
            computed[i + start] = true;
        }
    }

public:
    Obs(SCIP* scip) : scip(scip) {}

    inline double operator[](int index) {
        if (!computed[index]) {
            // Static cast completely removes the virtual vtable overhead
            static_cast<Derived*>(this)->compute(index);
        }
        return features[index];
    }

    void reset() {
        computed.fill(false); // Much faster than std::fill on vectors
    }

    void setVar(int probIndex) {
        if (probIndex >= 0) {
            var = getVarFromProbIndex(scip, probIndex);
        }
    }
};
#endif //OBS_H