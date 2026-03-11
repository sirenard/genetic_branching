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
    switch (index) {
        case 0:
            assign_features(gap(), index);
            break;
        case 1:
            assign_features(leafFrequency(), index);
            break;
        case 2:
            assign_features(treeWeight(), index);
            break;
        case 3:
            assign_features(completion(), index);
            break;
        case 4:
            assign_features(depth(), index);
            break;
        default:
            break;
    }
}


std::array<double, 1> TreeFeaturesObs::depth() {
    return {static_cast<double>(SCIPgetDepth(scip))};
}

std::array<double, 1> TreeFeaturesObs::gap() {
    return {SCIPgetGap(scip)};
}

std::array<double, 1> TreeFeaturesObs::leafFrequency() {
    double k = static_cast<double>(SCIPgetNNodes(scip));
    double fk = SCIPgetNLeaves(scip);
    return {safe_div<double>(fk - 0.5, k)};
}

std::array<double, 1> TreeFeaturesObs::treeWeight() {
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

    return {treeWeight};
}

std::array<double, 1> TreeFeaturesObs::completion() {
    double nnodes = static_cast<double>(SCIPgetNNodes(scip));
    double estimate = SCIPgetTreesizeEstimation(scip);
    return {safe_div<double>(nnodes, estimate)};
}

