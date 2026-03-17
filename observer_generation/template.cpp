#include "template_name.h"


#define FORMULA 0

#define FORMULA_STR ""

#ifdef USE_PYTHON
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
#endif


template_name::template_name(SCIP *scip, int priority)
      : ObjBranchrule(scip, "template_name", "Automatically generated", priority, -1,
                      1) {}

SCIP_DECL_BRANCHINITSOL(template_name::scip_initsol){
  tree_features = std::make_unique<TreeFeaturesObs>(scip);
  static_features.resize(SCIPgetNVars(scip));
  return SCIP_OKAY;
}

SCIP_DECL_BRANCHEXECLP(template_name::scip_execlp) {
  SCIP_VAR **lpcands;
  SCIP_Real* lpsols;
  int nlpcands;

  /* get branching candidates */
  SCIP_CALL(SCIPgetLPBranchCands(scip, &lpcands, &lpsols, NULL, NULL, &nlpcands,
                                 NULL));

  int bestcand = 0;
  SCIP_Real bestScore = SCIP_REAL_MIN;

  tree_features->reset();

  if(nlpcands > 1){
      DynamicFeaturesObs dynamic_feature(scip);
      for (int i = 0; i < nlpcands; i++) {
        auto cand = lpcands[i];

        int prob_index = SCIPvarGetProbindex(cand);

        if (!static_features[prob_index]) {
          static_features[prob_index] = std::make_unique<StaticFeaturesObs>(scip);
        }

        auto& static_feature = *static_features[prob_index];

        dynamic_feature.reset();
        dynamic_feature.setVar(prob_index);
        static_feature.setVar(prob_index);

        SCIP_Real score = FORMULA;
        // Tie-breaking using SCIP tolerances, fractionality, and objective
        if (i == 0 || SCIPisGT(scip, score, bestScore)) {
          bestScore = score;
          bestcand = i;
        } else if (SCIPisEQ(scip, score, bestScore)) {
          // Secondary tie-breaker: Pseudocosts
          SCIP_Real best_pscost_score = SCIPgetVarDPseudocostScore(scip, lpcands[bestcand], lpsols[bestcand], 0.2);
          SCIP_Real cand_pscost_score = SCIPgetVarDPseudocostScore(scip, cand, lpsols[i], 0.2);

          if (SCIPisGT(scip, cand_pscost_score, best_pscost_score)) {
            bestScore = score;
            bestcand = i;
          } else if (SCIPisEQ(scip, cand_pscost_score, best_pscost_score)) {
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

SCIP_DECL_BRANCHEXECPS(template_name::scip_execps) {
  *result = SCIP_DIDNOTRUN;
  return SCIP_OKAY;
}

SCIP_DECL_BRANCHEXITSOL(template_name::scip_exitsol){
  static_features.clear();
  return SCIP_OKAY;
}

void include_template_name(SCIP *scip, int priority) {
  SCIPincludeObjBranchrule(scip, new template_name(scip, priority), TRUE);
}


#ifdef USE_PYTHON
/** Creates and adds the custom branching rule to SCIP */
void add_branching(py::object py_scip) {
  // Extract SCIP* from PyCapsule
  void *scip_ptr = PyCapsule_GetPointer(py_scip.ptr(), "scip");
  if (!scip_ptr) {
    throw py::error_already_set();
  }

  SCIP *scip = static_cast<SCIP *>(scip_ptr);
  include_template_name(scip);
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
#endif