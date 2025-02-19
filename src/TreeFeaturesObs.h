//
// Created by simon on 14/02/25.
//

#ifndef TREEFEATURESOBS_H
#define TREEFEATURESOBS_H

#include <vector>

#include <scip/scip.h>

class TreeFeaturesObs{
    SCIP* scip;

    float gap();
    float leafFrequency();
    float openNodes();
    float ssg();
    float treeWeight();
    float completion();
public:
    explicit TreeFeaturesObs(long scipl);

    /**
     * Extract featurse about the B&B tree:
     * gap
     * leaf frequency
     * tree weight
     * estimation of the completion
     * @return Vector of 4 features
     */
    std::vector<float> computeFeatures();
};



#endif //TREEFEATURESOBS_H
