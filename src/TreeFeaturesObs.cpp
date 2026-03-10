//
// Created by simon on 14/02/25.
//

#include "TreeFeaturesObs.h"
#include <scip/event_estim.h>

#include "utils.h"

TreeFeaturesObs::TreeFeaturesObs(SCIP* scip): Obs(scip, size){}

TreeFeaturesObs::TreeFeaturesObs(py::object py_scip) : TreeFeaturesObs(
        static_cast<SCIP *>(PyCapsule_GetPointer(py_scip.ptr(), "scip"))
    ) {
    if (!scip) {
        throw py::error_already_set();
    }
}

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
    double k = static_cast<double>(SCIPgetNNodes(scip));
    double fk = SCIPgetNLeaves(scip);
    return safe_div<double>(fk - 0.5, k);
}

double TreeFeaturesObs::openNodes() {
}

double TreeFeaturesObs::ssg() {
}

double TreeFeaturesObs::treeWeight() {
    double treeWeight = 0;
    int nleaves = 0;
    SCIP_NODE **leaves = nullptr;

    SCIPgetLeaves(scip, &leaves, &nleaves);

    if (leaves != nullptr) {
        for (int i = 0; i < nleaves; i++) {
            SCIP_NODE *node = leaves[i];
            treeWeight += std::pow(2, -SCIPnodeGetDepth(node));
        }
    }

    return treeWeight;
}

double TreeFeaturesObs::completion() {
    double nnodes = static_cast<double>(SCIPgetNNodes(scip));
    double estimate = SCIPgetTreesizeEstimation(scip);
    return safe_div<double>(nnodes, estimate);
}

