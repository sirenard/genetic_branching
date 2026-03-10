#include <iostream>
#include <map>
#include <algorithm>

#include <objscip/objbranchrule.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <scip/scip.h>

#include "../../src/DynamicFeaturesObs.h"
#include "../../src/StaticFeaturesObs.h"
#include "../../src/TreeFeaturesObs.h"

#define FORMULA 0

#define FORMULA_STR ""

namespace py = pybind11;

class FeaturesWrapper {
  StaticFeaturesObs &staticFeatures;
  TreeFeaturesObs &treeFeatures;
  DynamicFeaturesObs &dynamicFeatures;

public:
  FeaturesWrapper(StaticFeaturesObs &staticFeatures,
                  TreeFeaturesObs &treeFeatures,
                  DynamicFeaturesObs &dynamicFeatures)
      : staticFeatures(staticFeatures), treeFeatures(treeFeatures),
        dynamicFeatures(dynamicFeatures) {}

  double operator[](int index) {
    if (index < staticFeatures.size) {
      return staticFeatures[index];
    }

    index -= staticFeatures.size;
    if (index < treeFeatures.size) {
      return treeFeatures[index];
    }

    index -= treeFeatures.size;
    if (index < dynamicFeatures.size) {
      return dynamicFeatures[index];
    }

    throw std::out_of_range("index out of range");
  }
};

class template_name : public scip::ObjBranchrule {
  std::vector<std::unique_ptr<StaticFeaturesObs>> static_features;
  std::unique_ptr<TreeFeaturesObs> tree_features;

public:
  template_name(SCIP *scip)
      : ObjBranchrule(scip, "template_name", "Automatically generated", 0, -1,
                      1) {}

  SCIP_DECL_BRANCHINITSOL(scip_initsol) override {
    tree_features = std::make_unique<TreeFeaturesObs>(scip);
    static_features.resize(SCIPgetNVars(scip));
    return SCIP_OKAY;
  }

  SCIP_DECL_BRANCHEXECLP(scip_execlp) override {
    SCIP_VAR **lpcands;
    int nlpcands;

    /* get branching candidates */
    SCIP_CALL(SCIPgetLPBranchCands(scip, &lpcands, NULL, NULL, NULL, &nlpcands,
                                   NULL));

    int bestcand = 0;
    SCIP_Real bestScore = SCIP_REAL_MIN;

    tree_features->reset();

    if(nlpcands > 1){
        for (int i = 0; i < nlpcands; i++) {
          auto cand = lpcands[i];

          int prob_index = SCIPvarGetProbindex(cand);

          if (!static_features[prob_index]) {
            static_features[prob_index] = std::make_unique<StaticFeaturesObs>(scip);
          }

          auto dynamic_feature = DynamicFeaturesObs(scip);
          auto& static_feature = *static_features[prob_index];

          dynamic_feature.reset();

          dynamic_feature.setVar(prob_index);
          static_feature.setVar(prob_index);

          FeaturesWrapper features(static_feature, *tree_features,
                                   dynamic_feature);

          SCIP_Real score = FORMULA;
          // FIXED: Tie-breaking using SCIP tolerances, fractionality, and objective
          if (i == 0 || SCIPisGT(scip, score, bestScore)) {
            bestScore = score;
            bestcand = i;
          } else if (SCIPisEQ(scip, score, bestScore)) {
            // Secondary tie-breaker: Fractionality closest to 0.5
            SCIP_Real best_frac = SCIPfrac(scip, SCIPvarGetLPSol(lpcands[bestcand]));
            SCIP_Real cand_frac = SCIPfrac(scip, SCIPvarGetLPSol(cand));
            SCIP_Real best_dist = std::abs(best_frac - 0.5);
            SCIP_Real cand_dist = std::abs(cand_frac - 0.5);

            if (SCIPisLT(scip, cand_dist, best_dist)) {
              bestScore = score;
              bestcand = i;
            } else if (SCIPisEQ(scip, cand_dist, best_dist)) {
              // Tertiary tie-breaker: Highest objective coefficient
              if (SCIPisGT(scip, SCIPvarGetObj(cand), SCIPvarGetObj(lpcands[bestcand]))) {
                bestScore = score;
                bestcand = i;
              }
            }
          }
        }
    }

    SCIP_CALL(SCIPbranchVar(scip, lpcands[bestcand], NULL, NULL, NULL));

    *result = SCIP_BRANCHED;
    return SCIP_OKAY;
  }

  SCIP_DECL_BRANCHEXECPS(scip_execps) override {
    *result = SCIP_DIDNOTRUN;
    return SCIP_OKAY;
  }

  SCIP_DECL_BRANCHEXITSOL(scip_exitsol) override {
    static_features.clear();
    return SCIP_OKAY;
  }
};

/** Creates and adds the custom branching rule to SCIP */
void add_branching(py::object py_scip) {
  // Extract SCIP* from PyCapsule
  void *scip_ptr = PyCapsule_GetPointer(py_scip.ptr(), "scip");
  if (!scip_ptr) {
    throw py::error_already_set();
  }

  SCIP *scip = static_cast<SCIP *>(scip_ptr);

  SCIPincludeObjBranchrule(scip, new template_name(scip), TRUE);
}

std::string to_str(){
    return FORMULA_STR;
}

PYBIND11_MODULE(template_name, m) {
  m.def("add_branching", &add_branching,
        "Adds custom branching rule to SCIP");
  m.def("to_str", &to_str,
        "Get the string formula");
}