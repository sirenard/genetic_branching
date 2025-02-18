//
// Created by simon on 14/02/25.
//

#ifndef DYNAMICFEATURESOBS_H
#define DYNAMICFEATURESOBS_H

#include <vector>

#include <scip/scip.h>

class DynamicFeaturesObs{
    SCIP* scip;
    SCIP_VAR* var {};
public:
    DynamicFeaturesObs(long scipl, int probIndex);

    std::vector<float> getPseudoCosts();

    ~DynamicFeaturesObs();
};


#endif //DYNAMICFEATURESOBS_H
