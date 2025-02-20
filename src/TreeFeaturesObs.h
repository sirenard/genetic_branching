//
// Created by simon on 14/02/25.
//

#ifndef TREEFEATURESOBS_H
#define TREEFEATURESOBS_H

#include <vector>

#include <scip/scip.h>

#include "Obs.h"

class TreeFeaturesObs: public Obs {
    SCIP* scip;

    double gap();
    double leafFrequency();
    double openNodes();
    double ssg();
    double treeWeight();
    double completion();
    double depth();


public:
    static const int size = 5;
    explicit TreeFeaturesObs(long scipl);

    void compute(int index) override;
};



#endif //TREEFEATURESOBS_H
