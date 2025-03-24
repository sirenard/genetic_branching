//
// Created by simon on 14/02/25.
//

#include "TreeFeaturesObs.h"
#include <scip/event_estim.h>

TreeFeaturesObs::TreeFeaturesObs(long scipl): Obs(scipl, size){}

void TreeFeaturesObs::compute(int index) {
    double value;
    switch (index) {
        case 0:
            value = gap();
            break;
        case 1:
            value = leafFrequency();
            break;
        case 2:
            value = treeWeight();
            break;
        case 3:
            value = completion();
            break;
        case 4:
            value = depth();
            break;
        default:
            value = 0;
    }

    features[index] = value;
    computed[index] = true;
}


double TreeFeaturesObs::depth() {
    return SCIPgetDepth(scip);
}

double TreeFeaturesObs::gap() {
    return SCIPgetGap(scip);
}

double TreeFeaturesObs::leafFrequency() {
    int k = SCIPgetNNodes(scip);
    int fk = SCIPgetNLeaves(scip);
    return 1.0 / static_cast<double>(k) * (static_cast<double>(fk) - 0.5);
}

double TreeFeaturesObs::openNodes() {
}

double TreeFeaturesObs::ssg() {
}

double TreeFeaturesObs::treeWeight() {
    double treeWeight = 0;

    int nleaves = SCIPgetNLeaves(scip);
    SCIP_NODE **leaves = new SCIP_NODE *[nleaves];

    SCIPgetLeaves(scip, &leaves, &nleaves);

    for (int i = 0; i < nleaves; i++) {
        SCIP_NODE *node = leaves[i];
        treeWeight += std::pow(2, -SCIPnodeGetDepth(node));
    }

    return treeWeight;
}

double TreeFeaturesObs::completion() {
    int nnodes = SCIPgetNNodes(scip);
    double estimate = SCIPgetTreesizeEstimation(scip);

    return static_cast<double>(nnodes) / estimate;
}

