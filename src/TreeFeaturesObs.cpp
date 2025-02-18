//
// Created by simon on 14/02/25.
//

#include "TreeFeaturesObs.h"
#include <scip/event_estim.h>

TreeFeaturesObs::TreeFeaturesObs(long scipl): scip(reinterpret_cast<SCIP *>(scipl))  {}

float TreeFeaturesObs::gap() {
    return SCIPgetGap(scip);
}

float TreeFeaturesObs::leafFrequency() {
    int k = SCIPgetNNodes(scip);
    int fk = SCIPgetNLeaves(scip);
    return 1.0/static_cast<float>(k) * (static_cast<float>(fk) - 0.5);
}

float TreeFeaturesObs::openNodes() {

}

float TreeFeaturesObs::ssg() {

}

float TreeFeaturesObs::treeWeight() {
    float treeWeight = 0;

    int nleaves = SCIPgetNLeaves(scip);
    SCIP_NODE** leaves = new SCIP_NODE*[nleaves];

    SCIPgetLeaves(scip, &leaves, &nleaves);

    for (int i = 0; i < nleaves; i++) {
        SCIP_NODE* node = leaves[i];
        treeWeight += std::pow(2,-SCIPnodeGetDepth(node));
    }

    return treeWeight;
}

float TreeFeaturesObs::completion() {
    int nnodes = SCIPgetNNodes(scip);
    double estimate = SCIPgetTreesizeEstimation(scip);

    return static_cast<float>(nnodes) / static_cast<float>(estimate);
}

std::vector<float> TreeFeaturesObs::computeFeatures() {
    return {
        gap(),
        leafFrequency(),
        treeWeight(),
        completion(),
    };
}
